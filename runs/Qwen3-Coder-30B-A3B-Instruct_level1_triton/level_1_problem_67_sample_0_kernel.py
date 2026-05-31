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
    dilation,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_BLOCK_SIZE: tl.constexpr,
    CHANNELS_BLOCK_SIZE: tl.constexpr,
    OUTPUT_LENGTH_BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    channel_id = tl.program_id(2)
    output_length_id = tl.program_id(3)
    
    # Calculate offsets
    batch_offset = batch_id * in_channels * input_length
    group_offset = group_id * (in_channels // groups) * input_length
    channel_offset = channel_id * output_length
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_BLOCK_SIZE, OUTPUT_LENGTH_BLOCK_SIZE + 2 * padding))
    
    # Load weights
    weight_offset = channel_id * (in_channels // groups) * kernel_size
    weight = tl.load(weight_ptr + weight_offset + tl.arange(0, kernel_size))
    
    # Bias
    bias = tl.load(bias_ptr + channel_id) if bias_ptr is not None else 0.0
    
    # Process output positions
    output_start = output_length_id * OUTPUT_LENGTH_BLOCK_SIZE
    for output_pos in range(output_start, min(output_start + OUTPUT_LENGTH_BLOCK_SIZE, output_length)):
        # Load input window into shared memory
        input_start = output_pos * stride - padding
        for i in range(GROUPS_BLOCK_SIZE):
            if i < (in_channels // groups):
                input_idx = input_start + i * dilation
                if input_idx >= 0 and input_idx < input_length:
                    shared_input[i, output_pos - output_start + padding] = tl.load(input_ptr + batch_offset + group_offset + input_idx)
                else:
                    shared_input[i, output_pos - output_start + padding] = 0.0
        
        # Compute convolution
        acc = 0.0
        for k in range(kernel_size):
            for i in range(GROUPS_BLOCK_SIZE):
                if i < (in_channels // groups):
                    acc += shared_input[i, output_pos - output_start + padding + k] * weight[k]
        
        # Store result
        output_idx = batch_id * out_channels * output_length + channel_id * output_length + output_pos
        tl.store(output_ptr + output_idx, acc + bias)

def triton_conv1d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Configure block sizes
    BLOCK_SIZE = 128
    GROUPS_BLOCK_SIZE = 32
    CHANNELS_BLOCK_SIZE = 32
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
        dilation,
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        """
        # Ensure inputs are on correct device and contiguous
        x = x.contiguous()
        
        # Use Triton kernel for convolution
        if self.bias is not None:
            return triton_conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return triton_conv1d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)