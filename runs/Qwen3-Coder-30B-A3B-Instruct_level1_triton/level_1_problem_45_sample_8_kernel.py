import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    
    # Calculate output dimensions
    output_x = tl.arange(0, BLOCK_SIZE)
    
    # Calculate input coordinates
    input_y_start = output_y * stride_h - padding_h
    input_x_start = output_x * stride_w - padding_w
    
    # Shared memory for reduction
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over kernel size
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            input_y = input_y_start + ky
            input_x = input_x_start + kx
            
            # Check bounds
            valid_y = (input_y >= 0) & (input_y < input_height)
            valid_x = (input_x >= 0) & (input_x < input_width)
            valid = valid_y & valid_x
            
            # Calculate input index
            input_idx = (
                batch_idx * (channels * input_height * input_width) +
                channel_idx * (input_height * input_width) +
                input_y * input_width +
                input_x
            )
            
            # Load input value if valid
            input_val = tl.load(input_ptr + input_idx, mask=valid, other=0.0)
            accumulator += input_val
    
    # Calculate average
    divisor = kernel_h * kernel_w
    output_val = accumulator / divisor
    
    # Store output
    output_idx = (
        batch_idx * (channels * output_height * output_width) +
        channel_idx * (output_height * output_width) +
        output_y * output_width +
        output_x
    )
    
    # Only store if within bounds
    valid_output = (output_x < output_width)
    tl.store(output_ptr + output_idx, output_val, mask=valid_output)

def triton_avg_pool2d(
    x: torch.Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int
):
    """
    Triton implementation of 2D Average Pooling
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dtype == torch.float32, "Input tensor must be FP32."
    
    batch_size, channels, input_height, input_width = x.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding_h - kernel_h) // stride_h + 1
    output_width = (input_width + 2 * padding_w - kernel_w) // stride_w + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, channels, output_height, output_width, device=x.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        channels,
        (output_height + 127) // 128  # Adjust block size as needed
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    avg_pool2d_kernel[grid](
        x,
        out,
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 2D Average Pooling.
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
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool2d(
            x,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding
        )

# For compatibility with existing test functions
def get_inputs():
    batch_size = 16
    channels = 64
    height = 2048
    width = 2048
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    kernel_size = 11
    return [kernel_size]