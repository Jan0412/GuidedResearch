import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_row = tl.program_id(2)
    output_col = tl.program_id(3)
    
    # Calculate global output position
    output_idx = batch_id * (in_channels * output_height * output_width) + \
                 channel_id * (output_height * output_width) + \
                 output_row * output_width + output_col
    
    # Shared memory for input tile and kernel
    input_tile = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    kernel_tile = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate kernel center offset
    kernel_center = kernel_size // 2
    
    # Loop over kernel elements
    for k_row in range(kernel_size):
        for k_col in range(kernel_size):
            # Calculate input coordinates
            input_row = output_row * stride + k_row - padding
            input_col = output_col * stride + k_col - padding
            
            # Check bounds
            if (input_row >= 0 and input_row < input_height and 
                input_col >= 0 and input_col < input_width):
                
                # Calculate input index
                input_idx = batch_id * (in_channels * input_height * input_width) + \
                           channel_id * (input_height * input_width) + \
                           input_row * input_width + input_col
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True)
                
                # Load kernel value
                kernel_val = tl.load(weight_ptr + channel_id * kernel_size * kernel_size + 
                                   k_row * kernel_size + k_col, mask=True)
                
                # Accumulate
                acc += input_val * kernel_val
    
    # Store result
    tl.store(output_ptr + output_idx, acc[0])

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
    # Grid dimensions
    grid = (
        batch_size,                    # batch dimension
        in_channels,                   # channel dimension  
        output_height,                 # output rows
        output_width                   # output cols
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.
    Optimized with Triton kernels for better performance.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weight tensor
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias_param', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Use Triton kernel for depthwise convolution
        output = triton_depthwise_conv2d(
            x, 
            self.weight, 
            stride=self.stride, 
            padding=self.padding
        )
        
        # Add bias if present
        if self.bias_param is not None:
            # Expand bias to match output shape
            bias_expanded = self.bias_param.view(1, -1, 1, 1)
            output = output + bias_expanded
            
        return output