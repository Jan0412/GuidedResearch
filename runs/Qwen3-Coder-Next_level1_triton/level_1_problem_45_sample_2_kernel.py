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
    # Get the batch, channel, and output position
    bc_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Compute input batch and channel indices
    b = bc_idx // channels
    c = bc_idx % channels
    
    # Compute the top-left corner of the pooling window in input space
    h_start = out_h_idx * stride_h - pad_h
    w_start = out_w_idx * stride_w - pad_w
    
    # Initialize accumulator
    acc = 0.0
    count = 0
    
    # Iterate over the pooling window
    for kh in range(kernel_h):
        h = h_start + kh
        # Check if within bounds (handling padding)
        if h >= 0 and h < in_h:
            for kw in range(kernel_w):
                w = w_start + kw
                # Check if within bounds (handling padding)
                if w >= 0 and w < in_w:
                    # Compute input pointer offset
                    offset = (b * channels * in_h * in_w + 
                             c * in_h * in_w + 
                             h * in_w + 
                             w)
                    val = tl.load(x_ptr + offset)
                    acc += val
                    count += 1
    
    # Compute average (avoid division by zero)
    if count > 0:
        avg = acc / count
    else:
        avg = 0.0
    
    # Store result
    out_offset = (b * channels * out_h * out_w + 
                 c * out_h * out_w + 
                 out_h_idx * out_w + 
                 out_w_idx)
    tl.store(out_ptr + out_offset, avg)


def triton_avg_pool2d(x, kernel_size, stride=None, padding=0):
    """
    Applies 2D Average Pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window (int or tuple)
        stride: Stride of the pooling operation (int or tuple, default: kernel_size)
        padding: Padding applied to the input (int or tuple, default: 0)
    
    Returns:
        Output tensor with Average Pooling applied
    """
    # Ensure input is on CUDA and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Extract dimensions
    batch_size, channels, in_h, in_w = x.shape
    
    # Handle kernel_size, stride, and padding
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size
        
    if stride is None:
        stride_h = stride_w = kernel_h
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    # Calculate output dimensions
    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    # Each block handles one (batch, channel) and one output position
    grid = (batch_size * channels, out_h, out_w)
    
    # Launch the Triton kernel
    avg_pool2d_kernel[grid](
        x, out,
        batch_size, channels, in_h, in_w,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_SIZE=128  # Not used in this implementation but kept for consistency
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