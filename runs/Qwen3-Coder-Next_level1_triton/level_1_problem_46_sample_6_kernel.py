import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avgpool3d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # B
    channels,  # C
    in_d, in_h, in_w,  # Input dimensions
    out_d, out_h, out_w,  # Output dimensions
    kernel_d, kernel_h, kernel_w,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Stride dimensions
    pad_d, pad_h, pad_w,  # Padding dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output tensor indices
    out_idx = tl.program_id(0)
    out_b = out_idx // (out_d * out_h * out_w)
    rest = out_idx % (out_d * out_h * out_w)
    out_d_idx = rest // (out_h * out_w)
    rest = rest % (out_h * out_w)
    out_h_idx = rest // out_w
    out_w_idx = rest % out_w
    
    # Compute input tensor starting indices
    in_d_start = out_d_idx * stride_d - pad_d
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w
    
    # Process multiple channels per block for efficiency
    c_start = tl.program_id(1) * BLOCK_SIZE
    c_mask = c_start + tl.arange(0, BLOCK_SIZE) < channels
    
    # Iterate over channels
    for c in range(channels):
        if c < c_start or c >= c_start + BLOCK_SIZE:
            continue
            
        # Compute average for this channel
        sum_val = 0.0
        count = 0
        
        # Iterate over kernel window
        for kd in range(kernel_d):
            in_d_pos = in_d_start + kd
            # Check if within input bounds (handle padding)
            if in_d_pos >= 0 and in_d_pos < in_d:
                for kh in range(kernel_h):
                    in_h_pos = in_h_start + kh
                    if in_h_pos >= 0 and in_h_pos < in_h:
                        for kw in range(kernel_w):
                            in_w_pos = in_w_start + kw
                            if in_w_pos >= 0 and in_w_pos < in_w:
                                # Compute input index
                                in_idx = (out_b * channels * in_d * in_h * in_w +
                                         c * in_d * in_h * in_w +
                                         in_d_pos * in_h * in_w +
                                         in_h_pos * in_w +
                                         in_w_pos)
                                # Load and accumulate
                                val = tl.load(x_ptr + in_idx)
                                sum_val += val
                                count += 1
        
        # Compute average and store
        if count > 0:
            avg_val = sum_val / count
        else:
            avg_val = 0.0
            
        # Compute output index
        out_idx = (out_b * channels * out_d * out_h * out_w +
                  c * out_d * out_h * out_w +
                  out_d_idx * out_h * out_w +
                  out_h_idx * out_w +
                  out_w_idx)
        
        tl.store(out_ptr + out_idx, avg_val)


def triton_avgpool3d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Apply 3D average pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size: Size of the pooling kernel
        stride: Stride of the pooling operation (default: kernel_size)
        padding: Padding to apply (default: 0)
        
    Returns:
        Output tensor after 3D average pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Extract dimensions
    batch_size, channels, in_d, in_h, in_w = x.shape
    
    # Set stride if not provided
    if stride is None:
        stride = kernel_size
        
    # Set padding if not provided
    pad_d = pad_h = pad_w = padding
    
    # Compute output dimensions
    out_d = (in_d + 2 * pad_d - kernel_size) // stride + 1
    out_h = (in_h + 2 * pad_h - kernel_size) // stride + 1
    out_w = (in_w + 2 * pad_w - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_d, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Configure grid
    total_out_elements = batch_size * out_d * out_h * out_w
    BLOCK_SIZE = 16  # Channels per block
    
    grid = lambda meta: (
        total_out_elements,
        (channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    )
    
    # Launch kernel
    avgpool3d_kernel[grid](
        x, out,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, kernel_size, kernel_size,
        stride, stride, stride,
        pad_d, pad_h, pad_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using custom Triton kernel for 3D average pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avgpool3d(x, self.kernel_size, self.stride, self.padding)