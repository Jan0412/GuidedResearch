import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer (batch, channels, height, width)
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    channels,  # Number of channels
    in_height,  # Input height
    in_width,  # Input width
    out_height,  # Output height
    out_width,  # Output width
    kernel_h: tl.constexpr,  # Kernel height
    kernel_w: tl.constexpr,  # Kernel width
    stride_h: tl.constexpr,  # Stride height
    stride_w: tl.constexpr,  # Stride width
    pad_h: tl.constexpr,  # Padding height
    pad_w: tl.constexpr,  # Padding width
    dil_h: tl.constexpr,  # Dilation height
    dil_w: tl.constexpr,  # Dilation width
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the index of the output element this program handles
    pid = tl.program_id(0)
    
    # Each block handles one output element
    # Output index layout: [batch, channel, out_h, out_w]
    # We'll flatten the batch*channels*height*width space
    
    # Compute output indices
    out_idx = pid
    out_w = out_idx % out_width
    out_idx //= out_width
    out_h = out_idx % out_height
    out_idx //= out_height
    c = out_idx % channels
    b = out_idx // channels
    
    # Compute the starting input position for this output position
    h_start = out_h * stride_h - pad_h
    w_start = out_w * stride_w - pad_w
    
    # Initialize max with -inf
    max_val = -tl.core.float32("inf")
    
    # Iterate over the kernel window
    for kh in range(kernel_h):
        h = h_start + kh * dil_h
        # Skip if out of bounds (with padding/dilation)
        if h >= 0 and h < in_height:
            for kw in range(kernel_w):
                w = w_start + kw * dil_w
                if w >= 0 and w < in_width:
                    # Calculate input index
                    in_idx = ((b * channels + c) * in_height + h) * in_width + w
                    val = tl.load(x_ptr + in_idx)
                    max_val = tl.maximum(max_val, val)
    
    # Store the result
    tl.store(out_ptr + pid, max_val)


def triton_maxpool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    """
    Applies 2D max pooling using a Triton kernel.
    
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
    
    batch_size, channels, in_height, in_width = x.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Number of output elements
    n_elements = batch_size * channels * out_height * out_width
    BLOCK_SIZE = 128  # We don't really need this for per-output-element kernel, but keeping for consistency
    
    # Grid is just the number of output elements
    grid = (n_elements,)
    
    # Launch the Triton kernel
    maxpool2d_kernel[grid](
        x, out,
        batch_size, channels,
        in_height, in_width,
        out_height, out_width,
        kernel_h=kernel_size, kernel_w=kernel_size,
        stride_h=stride, stride_w=stride,
        pad_h=padding, pad_w=padding,
        dil_h=dilation, dil_w=dilation,
        BLOCK_SIZE=BLOCK_SIZE,
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
        Applies optimized Max Pooling 2D to the input tensor.

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