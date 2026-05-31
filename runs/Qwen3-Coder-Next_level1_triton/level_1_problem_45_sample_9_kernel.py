import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    channels,  # Number of channels
    in_h, in_w,  # Input height and width
    out_h, out_w,  # Output height and width
    kernel_h, kernel_w,  # Kernel height and width
    stride_h, stride_w,  # Stride height and width
    pad_h, pad_w,  # Padding height and width
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Compute output position
    pid_b = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1) % BLOCK_SIZE_C  # Channel index within block
    pid_h = tl.program_id(1) // BLOCK_SIZE_C  # Output row index
    pid_w = tl.program_id(2)  # Output column index
    
    # Calculate the top-left corner of the pooling window in input space
    h_start = pid_h * stride_h - pad_h
    w_start = pid_w * stride_w - pad_w
    
    # Compute the valid range of the pooling window
    h_start_clamped = tl.maximum(h_start, 0)
    w_start_clamped = tl.maximum(w_start, 0)
    h_end_clamped = tl.minimum(h_start + kernel_h, in_h)
    w_end_clamped = tl.minimum(w_start + kernel_w, in_w)
    
    # Calculate the effective window size (important for edge cases with padding)
    effective_kernel_h = h_end_clamped - h_start_clamped
    effective_kernel_w = w_end_clamped - w_start_clamped
    
    # Accumulator for the sum
    sum_val = 0.0
    
    # Iterate over the pooling window
    for kh in range(effective_kernel_h):
        h_idx = h_start_clamped + kh
        for kw in range(effective_kernel_w):
            w_idx = w_start_clamped + kw
            # Compute input index
            input_idx = (pid_b * channels * in_h * in_w + 
                        pid_c * in_h * in_w + 
                        h_idx * in_w + 
                        w_idx)
            x = tl.load(x_ptr + input_idx)
            sum_val += x
    
    # Compute average
    denom = effective_kernel_h * effective_kernel_w
    out_val = sum_val / denom
    
    # Store output
    out_idx = (pid_b * channels * out_h * out_w + 
              pid_c * out_h * out_w + 
              pid_h * out_w + 
              pid_w)
    tl.store(out_ptr + out_idx, out_val)


def triton_avg_pool2d(x, kernel_size, stride=None, padding=0):
    """
    Applies 2D Average Pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window (int or tuple)
        stride: Stride of the pooling operation (int or tuple, None means same as kernel_size)
        padding: Padding applied to the input (int or tuple)
    """
    # Handle tuple arguments
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h = kernel_w = kernel_size
        
    if stride is None:
        stride_h = stride_w = kernel_h
    elif isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride
        
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding
    
    # Get input dimensions
    batch_size, channels, in_h, in_w = x.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Set up grid dimensions
    # Grid: (batch_size, channels, out_h * out_w)
    # We'll use BLOCK_SIZE_C to group channels for better parallelism
    BLOCK_SIZE_C = 16  # Tunable parameter for channel grouping
    
    # Calculate number of channel blocks
    num_c_blocks = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Grid for the kernel
    grid = lambda meta: (
        batch_size,
        num_c_blocks * out_h,
        out_w
    )
    
    # Launch kernel
    avg_pool2d_kernel[grid](
        x, out,
        batch_size, channels,
        in_h, in_w,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_SIZE_H=1,  # We process one output row at a time for simplicity
        BLOCK_SIZE_W=1,  # We process one output column at a time for simplicity
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        num_warps=4,
        num_stages=3,
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