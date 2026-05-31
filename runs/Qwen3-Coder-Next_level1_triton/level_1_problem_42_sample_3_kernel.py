import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of output elements
    # Shape information
    batch_size, channels, in_h, in_w,
    out_h, out_w,
    kernel_size, stride, padding, dilation,
    # Block sizes for tiling
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    # Window dimensions
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
):
    # Compute output spatial coordinates
    pid_c = tl.program_id(0)  # batch_idx * channels + channel_idx
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Calculate batch and channel indices
    batch_idx = pid_c // channels
    channel_idx = pid_c % channels
    
    # Calculate input position
    h_start = pid_h * stride - padding
    w_start = pid_w * stride - padding
    
    # Offset for this batch and channel in input
    input_offset = batch_idx * channels * in_h * in_w + channel_idx * in_h * in_w
    
    # Initialize max value
    max_val = -float('inf')
    
    # Iterate over the pooling window
    for kh in range(KERNEL_H):
        h = h_start + kh * dilation
        for kw in range(KERNEL_W):
            w = w_start + kw * dilation
            
            # Check bounds
            if 0 <= h < in_h and 0 <= w < in_w:
                # Calculate input index
                input_idx = input_offset + h * in_w + w
                val = tl.load(x_ptr + input_idx)
                max_val = tl.maximum(max_val, val)
    
    # Store result
    out_idx = pid_c * out_h * out_w + pid_h * out_w + pid_w
    tl.store(out_ptr + out_idx, max_val)


def triton_maxpool2d(x, kernel_size, stride, padding, dilation):
    """
    Triton implementation of MaxPool2d.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, in_h, in_w = x.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    grid = (batch_size * channels, out_h, out_w)
    
    # Launch kernel
    maxpool2d_kernel[grid](
        x, out,
        batch_size * channels * out_h * out_w,
        batch_size, channels, in_h, in_w,
        out_h, out_w,
        kernel_size, stride, padding, dilation,
        BLOCK_H=1, BLOCK_W=1,
        KERNEL_H=kernel_size, KERNEL_W=kernel_size,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

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
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)