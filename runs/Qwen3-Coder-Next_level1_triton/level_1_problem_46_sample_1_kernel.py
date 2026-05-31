import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool3d_kernel(
    x_ptr,  # Input tensor pointer
    y_ptr,  # Output tensor pointer
    B, C, D_in, H_in, W_in,  # Input dimensions
    D_out, H_out, W_out,  # Output dimensions
    kernel_d, kernel_h, kernel_w,  # Kernel dimensions
    stride_d, stride_h, stride_w,  # Stride dimensions
    pad_d, pad_h, pad_w,  # Padding dimensions
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_B: tl.constexpr,
):
    # Compute output batch index
    batch_idx = tl.program_id(0)
    # Compute output channel index
    channel_idx = tl.program_id(1)
    
    # Compute output position in D, H, W dimensions
    out_d_base = tl.program_id(2)
    out_h_base = tl.program_id(3)
    out_w_base = tl.program_id(4)
    
    # Offset for the output position
    out_offset = (
        batch_idx * (C * D_out * H_out * W_out) +
        channel_idx * (D_out * H_out * W_out) +
        out_d_base * (H_out * W_out) +
        out_h_base * W_out +
        out_w_base
    )
    
    # Calculate the starting position in input for this output
    in_d_start = out_d_base * stride_d - pad_d
    in_h_start = out_h_base * stride_h - pad_h
    in_w_start = out_w_base * stride_w - pad_w
    
    # Accumulator for the sum
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate over the kernel window
    for kd in range(kernel_d):
        in_d = in_d_start + kd
        d_mask = (in_d >= 0) & (in_d < D_in)
        
        for kh in range(kernel_h):
            in_h = in_h_start + kh
            h_mask = (in_h >= 0) & (in_h < H_in)
            
            for kw in range(kernel_w):
                in_w = in_w_start + kw
                w_mask = (in_w >= 0) & (in_w < W_in)
                
                # Combine masks
                valid_mask = d_mask & h_mask & w_mask
                
                # Calculate input offset
                in_offset = (
                    batch_idx * (C * D_in * H_in * W_in) +
                    channel_idx * (D_in * H_in * W_in) +
                    in_d * (H_in * W_in) +
                    in_h * W_in +
                    in_w
                )
                
                # Load input value if valid, else 0
                val = tl.load(x_ptr + in_offset, mask=valid_mask, other=0.0)
                sum_val += val * tl.cast(valid_mask, tl.float32)
                count += tl.cast(valid_mask, tl.int32)
    
    # Compute average
    count_inv = tl.cast(1.0 / tl.maximum(count, 1), tl.float32)
    out_val = sum_val * count_inv
    
    # Store result
    tl.store(y_ptr + out_offset, out_val)

def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Apply 3D average pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, depth, height, width)
        kernel_size: Size of the pooling kernel
        stride: Stride of the pooling operation (defaults to kernel_size)
        padding: Padding to apply before pooling
        
    Returns:
        Output tensor after 3D average pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get input dimensions
    B, C, D_in, H_in, W_in = x.shape
    
    # Set stride if not provided
    if stride is None:
        stride = kernel_size
    
    # Calculate output dimensions
    D_out = (D_in + 2 * padding - kernel_size) // stride + 1
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((B, C, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    # We'll use a block size of 1 for B and C since the kernel handles them in the program_id
    # For spatial dimensions, we'll use reasonable block sizes
    BLOCK_D = 1
    BLOCK_H = 1
    BLOCK_W = 1
    BLOCK_C = 1
    BLOCK_B = 1
    
    # Grid dimensions: (batch, channel, D_out, H_out, W_out)
    grid = (B, C, D_out, H_out, W_out)
    
    # Launch the kernel
    avg_pool3d_kernel[grid](
        x, out,
        B, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size, kernel_size, kernel_size,
        stride, stride, stride,
        padding, padding, padding,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C, BLOCK_B=BLOCK_B,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model that uses custom Triton kernels for 3D average pooling.
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
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)