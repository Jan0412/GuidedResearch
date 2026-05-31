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
    output_x = tl.program_id(3)
    
    # Shared memory for storing the pooling window values
    shared_vals = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Initialize maximum value
    max_val = tl.full([], -float('inf'), dtype=tl.float32)
    
    # Calculate starting position in input
    input_start_y = output_y * stride_h - padding_h
    input_start_x = output_x * stride_w - padding_w
    
    # Calculate actual kernel size considering dilation
    actual_kernel_h = (kernel_h - 1) * dilation_h + 1
    actual_kernel_w = (kernel_w - 1) * dilation_w + 1
    
    # Iterate through kernel elements
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input coordinates
            input_y = input_start_y + kh * dilation_h
            input_x = input_start_x + kw * dilation_w
            
            # Check bounds
            if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                # Calculate global index
                idx = batch_idx * (channels * input_height * input_width) + \
                      channel_idx * (input_height * input_width) + \
                      input_y * input_width + input_x
                
                # Load value
                val = tl.load(input_ptr + idx, mask=True)
                max_val = tl.maximum(max_val, val)
    
    # Write result
    if output_y < output_height and output_x < output_width:
        output_idx = batch_idx * (channels * output_height * output_width) + \
                     channel_idx * (output_height * output_width) + \
                     output_y * output_width + output_x
        tl.store(output_ptr + output_idx, max_val)

def triton_maxpool2d(input_tensor, kernel_size, stride, padding, dilation):
    """
    Triton implementation of 2D Max Pooling
    """
    batch_size, channels, height, width = input_tensor.shape
    
    # Calculate output dimensions
    output_height = (height + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    output_width = (width + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Handle case where kernel size is 1
    if kernel_size == 1 and stride == 1 and padding == 0 and dilation == 1:
        # Direct copy since it's just identity operation
        return input_tensor
    
    # Grid configuration
    grid = (
        batch_size,
        channels,
        output_height,
        output_width
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
        BLOCK_SIZE=1024
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Max Pooling 2D.
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
        Applies optimized Max Pooling 2D to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        return triton_maxpool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)