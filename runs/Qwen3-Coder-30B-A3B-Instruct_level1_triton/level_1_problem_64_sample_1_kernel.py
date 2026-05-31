import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv1d_transpose_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_size,
    weight_size,
    output_size,
    batch_size,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_pos_id = tl.program_id(2)
    
    # Calculate output position
    output_pos = out_pos_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_pos < output_size
    
    # Shared memory for input chunk
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(GROUP_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific indices
        group_in_ch = in_channels // groups
        group_out_ch = out_channels // groups
        group_input_offset = batch_id * in_channels * input_size + g * group_in_ch * input_size
        group_weight_offset = g * group_out_ch * group_in_ch * kernel_size
        group_output_offset = batch_id * out_channels * output_size + out_ch_id * output_size
        
        # Loop over kernel positions
        for k in range(kernel_size):
            # Calculate input position
            input_pos = output_pos - padding + k * stride
            
            # Load input chunk (with masking)
            input_chunk = tl.load(input_ptr + group_input_offset + 
                                tl.arange(0, BLOCK_SIZE) * input_size + 
                                input_pos, mask=input_pos >= 0 & input_pos < input_size, other=0.0)
            
            # Load weight
            weight_val = tl.load(weight_ptr + group_weight_offset + 
                               (out_ch_id % group_out_ch) * group_in_ch * kernel_size + 
                               (out_ch_id // group_out_ch) * group_in_ch * kernel_size + 
                               k)
            
            # Accumulate
            acc += input_chunk * weight_val
    
    # Store result
    tl.store(output_ptr + group_output_offset + output_pos, acc, mask=mask)

def triton_conv1d_transpose(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d operation
    """
    batch_size, in_channels, input_size = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output size
    output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_size, device=input_tensor.device, dtype=torch.float32)
    
    # Define grid dimensions
    grid = (
        batch_size,
        out_channels,
        (output_size + 127) // 128
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    GROUP_SIZE = 32
    
    conv1d_transpose_kernel[grid](
        input_tensor,
        weight,
        output,
        input_size,
        weight.shape[0] * weight.shape[1] * weight.shape[2],
        output_size,
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )