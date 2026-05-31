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
    
    # Calculate global output index
    output_idx = batch_idx * channels * output_height * output_width + \
                 channel_idx * output_height * output_width + \
                 output_y * output_width
    
    # Shared memory for reduction
    shared_max = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Loop over output width
    for output_x in range(output_width):
        # Initialize max value
        max_val = tl.full([], float(-float('inf')), dtype=tl.float32)
        
        # Loop over kernel elements
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input coordinates
                ih = output_y * stride_h - padding_h + kh * dilation_h
                iw = output_x * stride_w - padding_w + kw * dilation_w
                
                # Check if within bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Calculate input index
                    input_idx = batch_idx * channels * input_height * input_width + \
                                channel_idx * input_height * input_width + \
                                ih * input_width + iw
                    
                    # Load input value
                    val = tl.load(input_ptr + input_idx, mask=True, other=float('-inf'))
                    max_val = tl.maximum(max_val, val)
        
        # Store result
        tl.store(output_ptr + output_idx + output_x, max_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for Max Pooling 2D.
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
        Applies optimized Max Pooling 2D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D, shape (batch_size, channels, pooled_height, pooled_width).
        """
        batch_size, channels, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        output_width = (input_width + 2 * self.padding - (self.dilation * (self.kernel_size - 1) + 1)) // self.stride + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Grid configuration
        grid = (
            batch_size,       # batch dimension
            channels,         # channel dimension  
            output_height     # output height dimension
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
            BLOCK_SIZE=1024
        )
        
        return output