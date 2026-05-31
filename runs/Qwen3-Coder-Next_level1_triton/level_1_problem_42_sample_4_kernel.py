import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer
    y_ptr,  # Output tensor pointer
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
    pad_h,  # Padding height (top/bottom)
    pad_w,  # Padding width (left/right)
    dilation_h,  # Dilation height
    dilation_w,  # Dilation width
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output coordinates
    out_batch = tl.program_id(0)
    out_channel = tl.program_id(1)
    out_y = tl.program_id(2) // out_w
    out_x = tl.program_id(2) % out_w

    # Compute input starting position (top-left of the pooling window)
    in_y_start = out_y * stride_h - pad_h
    in_x_start = out_x * stride_w - pad_w

    # Initialize max value with very small number
    max_val = -float('inf')

    # Iterate over the pooling window
    for k_y in range(kernel_h):
        for k_x in range(kernel_w):
            # Compute actual input position with dilation
            in_y = in_y_start + k_y * dilation_h
            in_x = in_x_start + k_x * dilation_w

            # Check if within bounds
            if (0 <= in_y < in_h) and (0 <= in_x < in_w):
                # Compute input index
                in_index = (out_batch * channels * in_h * in_w +
                           out_channel * in_h * in_w +
                           in_y * in_w + in_x)
                val = tl.load(x_ptr + in_index)
                max_val = tl.maximum(max_val, val)

    # Compute output index
    out_index = (out_batch * channels * out_h * out_w +
                out_channel * out_h * out_w +
                out_y * out_w + out_x)
    tl.store(y_ptr + out_index, max_val)


def triton_maxpool2d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Triton implementation of MaxPool2d.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window
        padding: Padding to be applied before pooling
        dilation: Spacing between kernel elements
    
    Returns:
        Output tensor after max pooling
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, in_h, in_w = x.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    if out_h <= 0 or out_w <= 0:
        return out
    
    # Define block size and grid
    BLOCK_SIZE = 256
    # Grid: (batch_size, channels, out_h * out_w)
    grid = (batch_size, channels, out_h * out_w)
    
    # Launch kernel
    maxpool2d_kernel[grid](
        x, out,
        batch_size, channels,
        in_h, in_w,
        out_h, out_w,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        dilation, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Max Pooling 2D.
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