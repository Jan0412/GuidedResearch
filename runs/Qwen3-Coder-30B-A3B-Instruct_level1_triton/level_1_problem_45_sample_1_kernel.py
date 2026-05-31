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
    output_x = tl.program_id(3)
    
    # Calculate the starting position in the input tensor
    input_start_y = output_y * stride_h - padding_h
    input_start_x = output_x * stride_w - padding_w
    
    # Initialize accumulator
    sum_val = tl.zeros((1,), dtype=tl.float32)
    count = 0
    
    # Iterate through kernel
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            input_y = input_start_y + ky
            input_x = input_start_x + kx
            
            # Check bounds
            if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                # Calculate input index
                input_idx = batch_idx * (channels * input_height * input_width) + \
                           channel_idx * (input_height * input_width) + \
                           input_y * input_width + input_x
                
                # Load value and accumulate
                val = tl.load(input_ptr + input_idx, mask=True)
                sum_val += val
                count += 1
    
    # Calculate average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Calculate output index
    output_idx = batch_idx * (channels * output_height * output_width) + \
                channel_idx * (output_height * output_width) + \
                output_y * output_width + output_x
    
    # Store result
    tl.store(output_ptr + output_idx, avg_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer with Triton optimization.

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
        batch_size, channels, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Ensure output dimensions are valid
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Invalid output dimensions after pooling")
        
        # Prepare output tensor
        output = torch.empty(batch_size, channels, output_height, output_width, dtype=torch.float32, device=x.device)
        
        # Make input contiguous
        x = x.contiguous()
        
        # Define block size for Triton kernel
        BLOCK_SIZE = 128
        
        # Grid configuration
        grid = (
            batch_size,
            channels,
            output_height,
            output_width
        )
        
        # Launch kernel
        avg_pool2d_kernel[grid](
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
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output