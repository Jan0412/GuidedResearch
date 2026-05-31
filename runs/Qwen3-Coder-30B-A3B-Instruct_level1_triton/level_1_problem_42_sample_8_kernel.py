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
    # Get the batch and channel index for this program
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position in the output
    output_row_start = tl.program_id(2) * BLOCK_SIZE
    output_col_start = tl.program_id(3) * BLOCK_SIZE
    
    # Shared memory for the tile
    tile_size = BLOCK_SIZE * BLOCK_SIZE
    shared_tile = tl.shared_memory(dtype=tl.float32, size=tile_size)
    
    # Loop over the output dimensions
    for output_row in range(output_row_start, min(output_row_start + BLOCK_SIZE, output_height)):
        for output_col in range(output_col_start, min(output_col_start + BLOCK_SIZE, output_width)):
            # Initialize maximum value
            max_val = tl.full([], -float('inf'), dtype=tl.float32)
            
            # Iterate over kernel elements
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Calculate input coordinates with dilation and padding
                    input_row = output_row * stride_h - padding_h + kh * dilation_h
                    input_col = output_col * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input coordinates are valid
                    if (input_row >= 0 and input_row < input_height and 
                        input_col >= 0 and input_col < input_width):
                        
                        # Calculate input index
                        input_idx = (batch_idx * channels * input_height * input_width +
                                   channel_idx * input_height * input_width +
                                   input_row * input_width + input_col)
                        
                        # Load input value
                        val = tl.load(input_ptr + input_idx, mask=True, other=-float('inf'))
                        max_val = tl.maximum(max_val, val)
            
            # Write result to output
            output_idx = (batch_idx * channels * output_height * output_width +
                         channel_idx * output_height * output_width +
                         output_row * output_width + output_col)
            tl.store(output_ptr + output_idx, max_val)

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
        batch_size, channels, height, width = x.shape
        
        # Calculate output dimensions
        output_height = (height + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        output_width = (width + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Prepare kernel parameters
        kernel_h = self.kernel_size
        kernel_w = self.kernel_size
        stride_h = self.stride
        stride_w = self.stride
        padding_h = self.padding
        padding_w = self.padding
        dilation_h = self.dilation
        dilation_w = self.dilation
        
        # Ensure inputs are contiguous and on CUDA
        x = x.contiguous().to(torch.float32)
        output = output.contiguous()
        
        # Configure grid
        BLOCK_SIZE = 16
        grid = (
            batch_size,
            channels,
            (output_height + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (output_width + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
        
        # Launch kernel
        maxpool2d_kernel[grid](
            x,
            output,
            batch_size,
            channels,
            height,
            width,
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
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output