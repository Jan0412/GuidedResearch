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
    dilation,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate global indices
    batch_offset = batch_idx * in_channels * input_length
    channel_offset = channel_idx * input_length
    output_offset = output_idx * out_channels
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(CHANNELS_PER_BLOCK, kernel_size * dilation + 1))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUTS_PER_BLOCK,), dtype=tl.float32)
    
    # Process each output position
    for i in range(0, kernel_size):
        # Load weights for this kernel position
        weight_offsets = output_offset + tl.arange(0, OUTPUTS_PER_BLOCK)
        weight_values = tl.load(weight_ptr + channel_idx * kernel_size * out_channels + i * out_channels + tl.arange(0, OUTPUTS_PER_BLOCK))
        
        # Calculate input position
        input_pos = output_idx * stride + i * dilation
        
        # Load input values
        input_values = tl.zeros((CHANNELS_PER_BLOCK,), dtype=tl.float32)
        if input_pos >= 0 and input_pos < input_length:
            input_offsets = batch_offset + channel_offset + input_pos
            input_values = tl.load(input_ptr + input_offsets, mask=tl.arange(0, CHANNELS_PER_BLOCK) < in_channels, other=0.0)
        
        # Accumulate
        acc += tl.sum(weight_values * input_values, axis=0)
    
    # Add bias if enabled
    if bias_enabled:
        bias_values = tl.load(bias_ptr + output_offset, mask=tl.arange(0, OUTPUTS_PER_BLOCK) < out_channels, other=0.0)
        acc += bias_values
    
    # Store output
    output_offsets = batch_idx * out_channels * output_length + output_idx * out_channels + tl.arange(0, OUTPUTS_PER_BLOCK)
    tl.store(output_ptr + output_offsets, acc, mask=tl.arange(0, OUTPUTS_PER_BLOCK) < out_channels)

def triton_conv1d(input_tensor, weight, bias, stride, dilation):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * 0 - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define grid dimensions
    grid = (
        batch_size,                    # batch dimension
        (in_channels + 31) // 32,      # channel dimension (32 channels per block)
        output_length                  # output position dimension
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 32
    OUTPUTS_PER_BLOCK = 32
    
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
        dilation,
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUTS_PER_BLOCK=OUTPUTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)