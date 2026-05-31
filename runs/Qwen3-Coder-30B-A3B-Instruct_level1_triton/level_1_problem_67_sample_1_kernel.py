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
    GROUPS_BLOCK_SIZE: tl.constexpr,
    CHANNELS_BLOCK_SIZE: tl.constexpr,
    OUTPUT_LENGTH_BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    out_channel_id = tl.program_id(2)
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_BLOCK_SIZE, OUTPUT_LENGTH_BLOCK_SIZE + kernel_size - 1))
    
    # Calculate output position
    output_pos = tl.program_id(3) * OUTPUT_LENGTH_BLOCK_SIZE
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_LENGTH_BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate channel indices
        in_channel_start = g * (in_channels // groups)
        out_channel_start = out_channel_id * (out_channels // groups)
        
        # Load weights for this group and output channel
        weight_offset = out_channel_start * (in_channels // groups) * kernel_size + \
                       in_channel_start * kernel_size
        
        # Process each kernel element
        for k in range(kernel_size):
            # Load input window (with padding)
            input_offset = batch_id * in_channels * input_length + \
                          in_channel_start * input_length + \
                          output_pos * stride - padding
            
            # Copy input to shared memory
            for i in range(OUTPUT_LENGTH_BLOCK_SIZE + kernel_size - 1):
                if i < OUTPUT_LENGTH_BLOCK_SIZE:
                    input_idx = input_offset + i
                    if input_idx >= 0 and input_idx < input_length:
                        shared_input[g, i] = tl.load(input_ptr + input_idx, mask=True)
                    else:
                        shared_input[g, i] = 0.0
                else:
                    shared_input[g, i] = 0.0
            
            # Compute convolution for this kernel element
            for i in range(OUTPUT_LENGTH_BLOCK_SIZE):
                weight_idx = weight_offset + k
                acc[i] += shared_input[g, i + k] * tl.load(weight_ptr + weight_idx, mask=True)
                
    # Add bias if available
    if bias_ptr is not None:
        bias_offset = out_channel_id * (out_channels // groups)
        for i in range(OUTPUT_LENGTH_BLOCK_SIZE):
            acc[i] += tl.load(bias_ptr + bias_offset + i, mask=True)
            
    # Store output
    output_offset = batch_id * out_channels * output_length + \
                   out_channel_id * output_length + \
                   output_pos
    
    for i in range(OUTPUT_LENGTH_BLOCK_SIZE):
        tl.store(output_ptr + output_offset + i, acc[i], mask=output_pos + i < output_length)

def triton_conv1d(input_tensor, weight, bias, stride, padding, groups):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define block sizes
    BLOCK_SIZE = 128
    GROUPS_BLOCK_SIZE = 16
    CHANNELS_BLOCK_SIZE = 16
    OUTPUT_LENGTH_BLOCK_SIZE = 32
    
    # Launch grid
    grid = (
        batch_size,           # batch dimension
        groups,               # group dimension
        out_channels,         # output channel dimension
        (output_length + OUTPUT_LENGTH_BLOCK_SIZE - 1) // OUTPUT_LENGTH_BLOCK_SIZE  # output length dimension
    )
    
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
        GROUPS_BLOCK_SIZE=GROUPS_BLOCK_SIZE,
        CHANNELS_BLOCK_SIZE=CHANNELS_BLOCK_SIZE,
        OUTPUT_LENGTH_BLOCK_SIZE=OUTPUT_LENGTH_BLOCK_SIZE
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
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.padding, self.groups)