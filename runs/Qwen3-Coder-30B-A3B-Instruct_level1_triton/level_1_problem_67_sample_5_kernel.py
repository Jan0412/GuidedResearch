import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv1d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    OUTPUT_CHANNELS_PER_GROUP: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_pos_idx = tl.program_id(2)
    
    # Calculate group information
    group_idx = out_channel_idx // OUTPUT_CHANNELS_PER_GROUP
    local_channel_idx = out_channel_idx % OUTPUT_CHANNELS_PER_GROUP
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input positions for this kernel
    for k in range(kernel_size):
        # Calculate input position
        input_pos = out_pos_idx * stride + k - padding
        
        # Check bounds
        if input_pos >= 0 and input_pos < input_length:
            # Load input value
            input_val = tl.load(input_ptr + 
                              batch_idx * in_channels * input_length +
                              group_idx * CHANNELS_PER_GROUP * input_length +
                              (input_pos % input_length))
            
            # Load weight
            weight_val = tl.load(weight_ptr + 
                               out_channel_idx * kernel_size +
                               k)
            
            # Accumulate
            acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr != 0:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    # Store result
    if out_pos_idx < output_length:
        tl.store(output_ptr + 
                batch_idx * out_channels * output_length +
                out_channel_idx * output_length +
                out_pos_idx, 
                acc)

def triton_conv1d(input_tensor, weight, bias, stride=1, padding=0, groups=1):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_length
    )
    
    # Kernel parameters
    BLOCK_SIZE = 128
    GROUPS = groups
    CHANNELS_PER_GROUP = in_channels // groups
    OUTPUT_CHANNELS_PER_GROUP = out_channels // groups
    
    # Launch kernel
    conv1d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=GROUPS,
        CHANNELS_PER_GROUP=CHANNELS_PER_GROUP,
        OUTPUT_CHANNELS_PER_GROUP=OUTPUT_CHANNELS_PER_GROUP
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.padding, self.groups)