import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    channels,  # Number of channels
    height,  # Input height
    width,  # Input width
    out_height,  # Output height
    out_width,  # Output width
    kernel_size,  # Kernel size
    stride,  # Stride
    padding,  # Padding
    dilation,  # Dilation
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the batch and channel
    bc_id = tl.program_id(0)
    batch_id = bc_id // channels
    channel_id = bc_id % channels
    
    # Get the output position
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)
    
    # Calculate the top-left corner of the pooling window in the input
    h_start = out_h * stride - padding
    w_start = out_w * stride - padding
    
    # Initialize max with -inf
    max_val = -float('inf')
    
    # Iterate over the pooling window
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h = h_start + kh * dilation
            w = w_start + kw * dilation
            
            # Check if within bounds
            valid = (h >= 0) & (h < height) & (w >= 0) & (w < width)
            
            if valid:
                # Calculate input index
                input_idx = (
                    batch_id * channels * height * width +
                    channel_id * height * width +
                    h * width + w
                )
                
                # Load value and update max
                val = tl.load(x_ptr + input_idx)
                max_val = tl.maximum(max_val, val)
    
    # Store the result
    out_idx = (
        batch_id * channels * out_height * out_width +
        channel_id * out_height * out_width +
        out_h * out_width + out_w
    )
    tl.store(out_ptr + out_idx, max_val)


def triton_maxpool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    Applies 2D max pooling using a custom Triton kernel.
    
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
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Grid configuration: [batch_size * channels, out_height, out_width]
    grid = (batch_size * channels, out_height, out_width)
    
    # Launch the kernel
    BLOCK_SIZE = 1  # Not used in this kernel, but required for signature
    
    maxpool2d_kernel[grid](
        x, out,
        batch_size, channels, height, width,
        out_height, out_width,
        kernel_size, stride, padding, dilation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using Triton kernels.
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
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        return triton_maxpool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )