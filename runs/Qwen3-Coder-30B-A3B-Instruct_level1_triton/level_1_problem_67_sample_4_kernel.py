import torch
import torch.nn as nn
import torch.nn.functional as F
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
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_pos_idx = tl.program_id(2)
    
    # Calculate global position in output
    output_offset = batch_idx * out_channels * output_length + out_channel_idx * output_length + out_pos_idx
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate input start position for this output position
    input_start_pos = out_pos_idx * stride - padding
    
    # Group handling
    group_size = in_channels // groups
    group_idx = out_channel_idx // (out_channels // groups)
    channel_start = group_idx * group_size
    
    # Perform convolution
    for k in range(kernel_size):
        input_pos = input_start_pos + k
        if input_pos >= 0 and input_pos < input_length:
            # Calculate input offset
            input_offset = batch_idx * in_channels * input_length + channel_start * input_length + input_pos
            
            # Calculate weight offset  
            weight_offset = out_channel_idx * group_size * kernel_size + (k % group_size) * kernel_size + k
            
            # Load input value
            input_val = tl.load(input_ptr + input_offset, mask=(input_pos >= 0) & (input_pos < input_length))
            
            # Load weight value
            weight_val = tl.load(weight_ptr + weight_offset)
            
            # Accumulate
            acc += input_val * weight_val
    
    # Add bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + output_offset, acc)

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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        """
        batch_size, in_channels, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=torch.float32)
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Handle bias
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Define grid dimensions
        grid = (
            batch_size,
            self.out_channels,
            output_length
        )
        
        # Launch kernel
        BLOCK_SIZE = 128
        GROUP_SIZE = 8
        
        conv1d_kernel[grid](
            x,
            weight,
            output,
            bias_ptr,
            batch_size,
            in_channels,
            self.out_channels,
            input_length,
            output_length,
            self.kernel_size,
            self.stride,
            self.padding,
            self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE
        )
        
        return output

# Test code
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization