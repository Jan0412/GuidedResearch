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
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    
    # Calculate global output position
    output_x = tl.program_id(3)
    
    # Calculate input coordinates
    input_y_start = output_y * stride_h - padding_h
    input_x_start = output_x * stride_w - padding_w
    
    # Initialize maximum value
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    
    # Iterate through kernel
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            # Calculate input position with dilation
            input_y = input_y_start + ky * dilation_h
            input_x = input_x_start + kx * dilation_w
            
            # Check if within bounds
            if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                # Calculate input index
                input_idx = (
                    batch_idx * (channels * input_height * input_width) +
                    channel_idx * (input_height * input_width) +
                    input_y * input_width +
                    input_x
                )
                
                # Load input value
                val = tl.load(input_ptr + input_idx, mask=True, other=float('-inf'))
                max_val = tl.maximum(max_val, val)
    
    # Store result
    if output_y < output_height and output_x < output_width:
        output_idx = (
            batch_idx * (channels * output_height * output_width) +
            channel_idx * (output_height * output_width) +
            output_y * output_width +
            output_x
        )
        tl.store(output_ptr + output_idx, max_val[0])

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
        Applies optimized Max Pooling 2D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        batch_size, channels, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        output_width = (input_width + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid dimensions
        grid = (
            batch_size,
            channels,
            (output_height + 1) // 2,  # Use a reasonable block size for output_y
            (output_width + 1) // 2   # Use a reasonable block size for output_x
        )
        
        # Launch kernel
        maxpool2d_kernel[grid](
            x,
            output,
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

# Keep the original functions for compatibility
batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]