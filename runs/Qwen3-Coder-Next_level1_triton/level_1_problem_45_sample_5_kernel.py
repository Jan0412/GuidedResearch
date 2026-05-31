import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    channels,  # Number of channels
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
    # Get the output position
    out_batch = tl.program_id(0)
    out_c = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Calculate input position for top-left of the pooling window
    in_y_start = out_y * stride_h - pad_h
    in_x_start = out_x * stride_w - pad_w
    
    # Accumulator for the sum
    sum_val = 0.0
    count = 0
    
    # Iterate over the pooling window
    for k_y in range(kernel_h):
        for k_x in range(kernel_w):
            # Calculate actual input position
            in_y = in_y_start + k_y
            in_x = in_x_start + k_x
            
            # Check if within bounds (handle padding)
            if in_y >= 0 and in_y < in_h and in_x >= 0 and in_x < in_w:
                # Calculate input pointer offset
                offset = (out_batch * channels * in_h * in_w +
                         out_c * in_h * in_w +
                         in_y * in_w +
                         in_x)
                # Load value and add to sum
                val = tl.load(x_ptr + offset)
                sum_val += val
                count += 1
    
    # Compute average (only if we have valid elements)
    if count > 0:
        avg = sum_val / count
    else:
        avg = 0.0
    
    # Store result
    out_offset = (out_batch * channels * out_h * out_w +
                 out_c * out_h * out_w +
                 out_y * out_w +
                 out_x)
    tl.store(out_ptr + out_offset, avg)


def triton_avg_pool2d(x, kernel_size, stride=None, padding=0):
    """
    Applies 2D Average Pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window (int or tuple)
        stride: Stride of the pooling operation (int or tuple, default: kernel_size)
        padding: Padding applied to the input (int or tuple, default: 0)
    """
    # Handle tuple arguments
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size, kernel_size
    
    if stride is None:
        stride_h, stride_w = kernel_h, kernel_w
    elif isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride
        
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h, pad_w = padding, padding
    
    # Ensure input is contiguous and on CUDA
    x = x.contiguous()
    assert x.is_cuda, "Input tensor must be on CUDA device."
    
    # Get dimensions
    batch_size, channels, in_h, in_w = x.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions
    # We'll use a 4D grid: [batch, channels, out_h, out_w]
    # But for better performance, we'll group channels in blocks
    
    BLOCK_SIZE = 16  # Tunable parameter
    
    # Grid: [batch_size, channels, out_h, out_w]
    grid = (batch_size, channels, out_h, out_w)
    
    # Launch kernel
    avg_pool2d_kernel[grid](
        x, out,
        batch_size, channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using custom Triton kernel.
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