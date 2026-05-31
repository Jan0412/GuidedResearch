import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def maxpool2d_kernel(
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
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    
    # Calculate output dimensions
    output_x = tl.arange(0, BLOCK_SIZE)
    
    # Shared memory for local max pooling
    shared_max = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Initialize maximum value
    max_val = tl.full([], float(-float('inf')), dtype=tl.float32)
    
    # Loop over kernel elements
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input coordinates
            input_y = output_y * stride_h + kh * dilation_h - padding_h
            input_x = output_x * stride_w + kw * dilation_w - padding_w
            
            # Check bounds
            valid_mask = (input_y >= 0) & (input_y < input_height) & \
                        (input_x >= 0) & (input_x < input_width)
            
            # Load input data
            input_offset = batch_idx * (channels * input_height * input_width) + \
                          channel_idx * (input_height * input_width) + \
                          input_y * input_width + input_x
            
            # Load input values with masking
            input_vals = tl.load(input_ptr + input_offset, mask=valid_mask, other=float('-inf'))
            
            # Update maximum
            max_val = tl.maximum(max_val, input_vals)
    
    # Store results
    output_offset = batch_idx * (channels * output_height * output_width) + \
                   channel_idx * (output_height * output_width) + \
                   output_y * output_width + output_x
    
    tl.store(output_ptr + output_offset, max_val, mask=output_x < output_width)

def triton_maxpool2d(input_tensor, kernel_size, stride, padding, dilation):
    """
    Triton implementation of 2D Max Pooling
    """
    batch_size, channels, height, width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (height + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    output_width = (width + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Ensure output dimensions are positive
    output_height = max(1, output_height)
    output_width = max(1, output_width)
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Set up grid
    grid = (
        batch_size,
        channels,
        (output_height + 127) // 128  # Use 128 threads per block for output rows
    )
    
    # Launch kernel
    maxpool2d_kernel[grid](
        input_tensor,
        output,
        batch_size,
        channels,
        height,
        width,
        output_height,
        output_width,
        kernel_size,
        kernel_size,
        stride,
        stride,
        padding,
        padding,
        dilation,
        dilation,
        BLOCK_SIZE=128
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Max Pooling 2D.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer with Triton optimization.

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
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)