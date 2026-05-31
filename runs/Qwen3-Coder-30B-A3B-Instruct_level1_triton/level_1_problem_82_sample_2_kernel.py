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
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_row = tl.program_id(2)
    output_col = tl.program_id(3)
    
    # Calculate global output position
    output_idx = batch_idx * (in_channels * output_height * output_width) + \
                 channel_idx * (output_height * output_width) + \
                 output_row * output_width + output_col
    
    # Shared memory for input tile
    input_tile = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel elements
    for k_row in range(kernel_size):
        for k_col in range(kernel_size):
            # Calculate input coordinates
            input_row = output_row * stride + k_row - padding
            input_col = output_col * stride + k_col - padding
            
            # Check bounds
            if input_row >= 0 and input_row < input_height and input_col >= 0 and input_col < input_width:
                # Calculate input index
                input_idx = batch_idx * (in_channels * input_height * input_width) + \
                           channel_idx * (input_height * input_width) + \
                           input_row * input_width + input_col
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size + k_row * kernel_size + k_col)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + output_idx, acc[0])

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Ensure tensors are contiguous and on GPU
    input_tensor = input_tensor.contiguous().cuda()
    weight = weight.contiguous().cuda()
    
    # Allocate output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, dtype=torch.float32, device='cuda')
    
    # Define block sizes
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
    # Grid dimensions
    grid = (
        batch_size,
        in_channels,
        output_height,
        output_width
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
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Use our Triton implementation
        return triton_depthwise_conv2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding
        )

# Keep the original test code structure
def get_inputs():
    batch_size = 16
    in_channels = 64
    kernel_size = 3
    width = 512
    height = 512
    stride = 1
    padding = 0
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [64, 3, 1, 0]