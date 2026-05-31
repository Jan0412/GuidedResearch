import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,
    y_ptr,
    n_batches,
    n_channels,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Output spatial indices
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate input starting positions with padding
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w
    
    # Initialize max value to -inf
    max_val = -tl.float32("inf")
    
    # Iterate over the pooling window
    for kh in range(kernel_h):
        in_h_idx = in_h_start + kh * dil_h
        for kw in range(kernel_w):
            in_w_idx = in_w_start + kw * dil_w
            
            # Check if within input bounds
            if (0 <= in_h_idx < in_h) and (0 <= in_w_idx < in_w):
                # Calculate input pointer offset
                x_offset = (
                    batch_idx * n_channels * in_h * in_w +
                    channel_idx * in_h * in_w +
                    in_h_idx * in_w +
                    in_w_idx
                )
                val = tl.load(x_ptr + x_offset)
                max_val = tl.maximum(max_val, val)
    
    # Calculate output pointer offset
    y_offset = (
        batch_idx * n_channels * out_h * out_w +
        channel_idx * out_h * out_w +
        out_h_idx * out_w +
        out_w_idx
    )
    tl.store(y_ptr + y_offset, max_val)


def triton_maxpool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int
) -> torch.Tensor:
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
    
    batch_size, channels, in_h, in_w = x.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Define grid dimensions
    grid = (batch_size, channels, out_h, out_w)
    
    # Launch the Triton kernel
    maxpool2d_kernel[grid](
        x, out,
        batch_size, channels, in_h, in_w, out_h, out_w,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        dilation, dilation,
        BLOCK_SIZE=1
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
        Applies Max Pooling 2D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        return triton_maxpool2d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation
        )