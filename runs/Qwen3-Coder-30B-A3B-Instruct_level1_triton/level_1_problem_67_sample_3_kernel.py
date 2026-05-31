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
    GROUPS_BLOCK_SIZE: tl.constexpr,
    CHANNELS_BLOCK_SIZE: tl.constexpr,
    OUTPUT_LENGTH_BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_channel_idx = tl.program_id(2)
    
    # Shared memory for input tiles
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_BLOCK_SIZE, CHANNELS_BLOCK_SIZE, OUTPUT_LENGTH_BLOCK_SIZE + kernel_size - 1))
    
    # Calculate output position
    output_pos = tl.program_id(3) * OUTPUT_LENGTH_BLOCK_SIZE
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_LENGTH_BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific pointers
        input_group_offset = batch_idx * in_channels * input_length + g * (in_channels // groups) * input_length
        weight_group_offset = out_channel_idx * (in_channels // groups) * kernel_size + g * (in_channels // groups) * kernel_size
        bias_offset = out_channel_idx
        
        # Load bias if it exists
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + out_channel_idx)
            acc += bias_val
        
        # Process each channel in the group
        for c in range(in_channels // groups):
            # Load input tile into shared memory
            input_offset = input_group_offset + c * input_length
            
            # Load input data with padding
            for i in range(OUTPUT_LENGTH_BLOCK_SIZE + kernel_size - 1):
                if output_pos + i < input_length:
                    val = tl.load(input_ptr + input_offset + output_pos + i, mask=(output_pos + i < input_length), other=0.0)
                else:
                    val = 0.0
                tl.store(shared_input + g * (CHANNELS_BLOCK_SIZE * (OUTPUT_LENGTH_BLOCK_SIZE + kernel_size - 1)) + 
                        c * (OUTPUT_LENGTH_BLOCK_SIZE + kernel_size - 1) + i, val)
            
            # Compute convolution for this channel
            for k in range(kernel_size):
                weight_offset = weight_group_offset + c * kernel_size + k
                
                # Load weight
                weight_val = tl.load(weight_ptr + weight_offset)
                
                # Apply convolution
                for i in range(OUTPUT_LENGTH_BLOCK_SIZE):
                    if output_pos + i + k < input_length and output_pos + i >= 0:
                        input_val = tl.load(shared_input + g * (CHANNELS_BLOCK_SIZE * (OUTPUT_LENGTH_BLOCK_SIZE + kernel_size - 1)) + 
                                          c * (OUTPUT_LENGTH_BLOCK_SIZE + kernel_size - 1) + i + k)
                        acc[i] += weight_val * input_val
    
    # Store output
    output_offset = batch_idx * out_channels * output_length + out_channel_idx * output_length + output_pos
    for i in range(OUTPUT_LENGTH_BLOCK_SIZE):
        if output_pos + i < output_length:
            tl.store(output_ptr + output_offset + i, acc[i])

def triton_conv1d(input_tensor, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton-based 1D convolution implementation.
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    assert weight.is_cuda, "Weight tensor must be on CUDA"
    
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Configure block sizes
    BLOCK_SIZE = 128
    GROUPS_BLOCK_SIZE = 16
    CHANNELS_BLOCK_SIZE = 16
    OUTPUT_LENGTH_BLOCK_SIZE = 32
    
    # Grid configuration
    grid = (
        batch_size,
        groups,
        out_channels,
        (output_length + OUTPUT_LENGTH_BLOCK_SIZE - 1) // OUTPUT_LENGTH_BLOCK_SIZE
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        input_tensor.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
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
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS_BLOCK_SIZE=GROUPS_BLOCK_SIZE,
        CHANNELS_BLOCK_SIZE=CHANNELS_BLOCK_SIZE,
        OUTPUT_LENGTH_BLOCK_SIZE=OUTPUT_LENGTH_BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation using Triton optimization.
    """
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
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)