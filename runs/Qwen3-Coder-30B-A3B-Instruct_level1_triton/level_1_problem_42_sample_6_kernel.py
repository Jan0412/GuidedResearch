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
    shared_data = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Loop over output width
    for output_x in range(output_width):
        # Initialize max value
        max_val = tl.full([1], float('-inf'), dtype=tl.float32)
        
        # Loop over kernel
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input coordinates
                ih = output_y * stride_h - padding_h + kh * dilation_h
                iw = output_x * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Calculate input index
                    input_idx = batch_idx * channels * input_height * input_width + \
                                channel_idx * input_height * input_width + \
                                ih * input_width + iw
                    
                    # Load input value
                    val = tl.load(input_ptr + input_idx, mask=True)
                    max_val = tl.maximum(max_val, val)
        
        # Store result
        tl.store(output_ptr + output_idx + output_x, max_val)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Max Pooling 2D.
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
        
        # For simplicity, assuming square kernel for now
        if isinstance(kernel_size, int):
            self.kernel_h = kernel_size
            self.kernel_w = kernel_size
        else:
            self.kernel_h, self.kernel_w = kernel_size
            
        if isinstance(stride, int):
            self.stride_h = stride
            self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
            
        if isinstance(padding, int):
            self.padding_h = padding
            self.padding_w = padding
        else:
            self.padding_h, self.padding_w = padding
            
        if isinstance(dilation, int):
            self.dilation_h = dilation
            self.dilation_w = dilation
        else:
            self.dilation_h, self.dilation_w = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        batch_size, channels, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding_h - 
                        (self.dilation_h * (self.kernel_h - 1) + 1)) // self.stride_h + 1
        output_width = (input_width + 2 * self.padding_w - 
                       (self.dilation_w * (self.kernel_w - 1) + 1)) // self.stride_w + 1
        
        # Create output tensor
        output = torch.empty(batch_size, channels, output_height, output_width, 
                           dtype=torch.float32, device=x.device)
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Grid configuration
        grid = (
            batch_size,
            channels,
            output_height
        )
        
        # Block size for shared memory
        BLOCK_SIZE = 1024
        
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
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output

# Note: The above implementation has limitations in terms of performance due to 
# the current Triton API constraints. A more optimized version would require 
# better handling of shared memory and proper reduction operations across 
# different threads. However, this provides the basic structure for how 
# such a kernel could be implemented.