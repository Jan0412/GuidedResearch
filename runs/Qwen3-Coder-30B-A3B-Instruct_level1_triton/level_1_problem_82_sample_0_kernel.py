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
    output_row_id = tl.program_id(2)
    output_col_id = tl.program_id(3)
    
    # Calculate global output position
    output_idx = batch_id * (in_channels * output_height * output_width) + \
                 channel_id * (output_height * output_width) + \
                 output_row_id * output_width + output_col_id
    
    # Shared memory for input tile
    TILE_SIZE = kernel_size + 2 * padding
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(TILE_SIZE, TILE_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate input region boundaries
    start_h = output_row_id * stride - padding
    start_w = output_col_id * stride - padding
    
    # Load weights
    weight = tl.load(weight_ptr + channel_id * kernel_size * kernel_size)
    
    # Process kernel elements
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input coordinates
            h = start_h + k_h
            w = start_w + k_w
            
            # Check bounds
            if h >= 0 and h < input_height and w >= 0 and w < input_width:
                # Calculate input index
                input_idx = batch_id * (in_channels * input_height * input_width) + \
                           channel_id * (input_height * input_width) + \
                           h * input_width + w
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx)
                
                # Load weight
                weight_val = tl.load(weight_ptr + channel_id * kernel_size * kernel_size + k_h * kernel_size + k_w)
                
                # Accumulate
                acc += input_val * weight_val
            else:
                # Handle padding (zero padding)
                acc += 0.0 * tl.load(weight_ptr + channel_id * kernel_size * kernel_size + k_h * kernel_size + k_w)
    
    # Store result
    tl.store(output_ptr + output_idx, acc[0])

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[2]  # Assuming square kernel
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, dtype=torch.float32, device=input_tensor.device)
    
    # Configure grid
    grid = (
        batch_size,
        in_channels,
        output_height,
        output_width
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
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
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel for depthwise convolution
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)