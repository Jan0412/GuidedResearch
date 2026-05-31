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
    
    # Shared memory for the kernel
    kernel_shared = tl.shared_memory(dtype=tl.float32, size=kernel_size * kernel_size)
    
    # Load kernel weights into shared memory
    if tl.thread_id() < kernel_size * kernel_size:
        kernel_shared[tl.thread_id()] = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size + tl.thread_id())
    
    tl.sync()
    
    # Calculate output position
    output_pos = batch_idx * in_channels * output_height * output_width + \
                 channel_idx * output_height * output_width + \
                 output_row * output_width + output_col
    
    # Initialize accumulator
    acc = 0.0
    
    # Perform convolution
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Calculate input position
            input_row = output_row * stride - padding + kh
            input_col = output_col * stride - padding + kw
            
            # Check bounds
            if input_row >= 0 and input_row < input_height and \
               input_col >= 0 and input_col < input_width:
                input_pos = batch_idx * in_channels * input_height * input_width + \
                           channel_idx * input_height * input_width + \
                           input_row * input_width + input_col
                
                # Load input value
                input_val = tl.load(input_ptr + input_pos)
                
                # Load kernel value
                kernel_val = kernel_shared[kh * kernel_size + kw]
                
                # Accumulate
                acc += input_val * kernel_val
    
    # Store result
    tl.store(output_ptr + output_pos, acc)

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0):
    """
    Triton implementation of depthwise 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    kernel_size = weight.shape[-1]  # Assuming square kernel
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Ensure inputs are contiguous and on GPU
    input_tensor = input_tensor.contiguous().cuda()
    weight = weight.contiguous().cuda()
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, dtype=torch.float32, device='cuda')
    
    # Define block sizes
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 1
    OUTPUT_ELEMENTS_PER_BLOCK = 1
    
    # Grid dimensions
    grid = (
        batch_size,           # Batch dimension
        in_channels,          # Channel dimension
        output_height,        # Output height
        output_width          # Output width
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
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize weight tensor
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel for depthwise convolution
        output = triton_depthwise_conv2d(x, self.weight, self.stride, self.padding)
        
        # Add bias if present
        if self.bias is not None:
            # Bias is added per channel, so we need to reshape appropriately
            bias_reshaped = self.bias.view(1, -1, 1, 1)
            output = output + bias_reshaped
            
        return output