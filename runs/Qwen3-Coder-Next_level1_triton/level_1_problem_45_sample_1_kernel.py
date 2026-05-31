import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,  # Input tensor pointer (B, C, H, W)
    out_ptr,  # Output tensor pointer (B, C, out_h, out_w)
    n_batches,  # Batch size (B)
    n_channels,  # Number of channels (C)
    in_h,  # Input height
    in_w,  # Input width
    out_h,  # Output height
    out_w,  # Output width
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    pad_h,  # Padding height
    pad_w,  # Padding width
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate output position
    out_y = tl.program_id(3)
    
    # Calculate input position corresponding to output position
    in_y_start = out_y * stride_h - pad_h
    in_x_start = out_w_idx * stride_w - pad_w
    
    # Compute the sum and count of valid elements in the pooling window
    sum_val = 0.0
    count = 0
    
    # Iterate over the pooling window
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            in_y = in_y_start + kh
            in_x = in_x_start + kw
            
            # Check if within input bounds (handling padding)
            if (in_y >= 0) and (in_y < in_h) and (in_x >= 0) and (in_x < in_w):
                # Calculate input pointer offset
                offset = (batch_idx * n_channels * in_h * in_w + 
                         channel_idx * in_h * in_w + 
                         in_y * in_w + in_x)
                x_val = tl.load(x_ptr + offset)
                sum_val += x_val
                count += 1
    
    # Compute average (handle edge case where count might be 0 due to padding)
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Store result
    out_offset = (batch_idx * n_channels * out_h * out_w + 
                 channel_idx * out_h * out_w + 
                 out_y * out_w + out_w_idx)
    tl.store(out_ptr + out_offset, avg_val)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Applies 2D Average Pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling operation (default: kernel_size)
        padding: Padding applied to the input (default: 0)
    
    Returns:
        Output tensor after average pooling
    """
    # Ensure input is contiguous and on GPU
    x = x.contiguous()
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Parse parameters
    batch_size, channels, in_h, in_w = x.shape
    kernel_h = kernel_w = kernel_size
    stride_h = stride_w = stride if stride is not None else kernel_size
    pad_h = pad_w = padding
    
    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Define block size for parallelization
    BLOCK_SIZE = 16
    
    # Define grid dimensions: (batch, channel, out_w, out_h)
    grid = (batch_size, channels, out_w, out_h)
    
    # Launch kernel
    avg_pool2d_kernel[grid](
        x, out,
        batch_size, channels,
        in_h, in_w,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)