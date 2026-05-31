import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer (N, C, H, W)
    y_ptr,  # Output tensor pointer (N, C, out_h, out_w)
    n_elements,  # Total number of elements in output
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output tensor index
    pid = tl.program_id(0)
    if pid >= n_elements:
        return
    
    # Decode the output index into (n, c, oh, ow)
    # Layout: [batch_size, channels, out_h, out_w]
    tmp = pid
    ow = tmp % out_w
    tmp //= out_w
    oh = tmp % out_h
    tmp //= out_h
    c = tmp % channels
    n = tmp // channels
    
    # Calculate input position corresponding to (oh, ow)
    # The top-left corner of the pooling window in the input
    h_start = oh * stride - padding
    w_start = ow * stride - padding
    
    # Initialize max value to -inf
    max_val = -float('inf')
    
    # Iterate over the pooling window
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate actual input coordinates
            h = h_start + kh * dilation
            w = w_start + kw * dilation
            
            # Check bounds
            if h >= 0 and h < height and w >= 0 and w < width:
                # Calculate input index
                input_idx = ((n * channels + c) * height + h) * width + w
                val = tl.load(x_ptr + input_idx)
                max_val = tl.maximum(max_val, val)
    
    # Store the result
    tl.store(y_ptr + pid, max_val)


def triton_maxpool2d(x, kernel_size, stride, padding, dilation):
    """
    Applies 2D max pooling using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        kernel_size, stride, padding, dilation: Pooling parameters
    
    Returns:
        Output tensor after max pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get input dimensions
    batch_size, channels, height, width = x.shape
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Total number of output elements
    n_elements = out.numel()
    BLOCK_SIZE = 128  # Tunable parameter
    
    # Determine grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    maxpool2d_kernel[grid](
        x, out, n_elements,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        out_h=out_h,
        out_w=out_w,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)