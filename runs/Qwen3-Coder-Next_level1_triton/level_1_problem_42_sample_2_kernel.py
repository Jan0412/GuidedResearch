import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer
    y_ptr,  # Output tensor pointer
    N,  # Batch size
    C,  # Number of channels
    H,  # Input height
    W,  # Input width
    out_h,  # Output height
    out_w,  # Output width
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID maps to output element
    pid = tl.program_id(0)
    
    # Calculate output indices
    batch_idx = pid // (C * out_h * out_w)
    channel_idx = (pid % (C * out_h * out_w)) // (out_h * out_w)
    out_h_idx = (pid % (out_h * out_w)) // out_w
    out_w_idx = pid % out_w
    
    # Calculate input starting position
    h_start = out_h_idx * stride_h - pad_h
    w_start = out_w_idx * stride_w - pad_w
    
    # Initialize max value to -inf
    max_val = -float('inf')
    
    # Iterate over kernel window
    for kh in range(kernel_h):
        h = h_start + kh * dil_h
        # Skip if outside valid input range
        if h >= 0 and h < H:
            for kw in range(kernel_w):
                w = w_start + kw * dil_w
                # Skip if outside valid input range
                if w >= 0 and w < W:
                    # Calculate input pointer offset
                    offset = batch_idx * C * H * W + channel_idx * H * W + h * W + w
                    val = tl.load(x_ptr + offset)
                    max_val = tl.maximum(max_val, val)
    
    # Store result
    tl.store(y_ptr + pid, max_val)


def triton_maxpool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Applies 2D max pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding to be applied before pooling
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor after max pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, height, width = x.shape
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Total number of output elements
    n_elements = batch_size * channels * out_h * out_w
    
    # Set block size (can be tuned for performance)
    BLOCK_SIZE = 256
    
    # Determine grid size
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    maxpool2d_kernel[grid](
        x, out,
        batch_size, channels, height, width, out_h, out_w,
        kernel_size, kernel_size,  # kernel_h, kernel_w
        stride, stride,             # stride_h, stride_w
        padding, padding,           # pad_h, pad_w
        dilation, dilation,         # dil_h, dil_w
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the optimized Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D using Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)