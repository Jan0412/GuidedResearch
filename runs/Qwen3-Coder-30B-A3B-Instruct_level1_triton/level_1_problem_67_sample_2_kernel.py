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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUTS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    output_id = tl.program_id(2)
    
    # Calculate output position
    output_pos = output_id * OUTPUTS_PER_BLOCK
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(KERNEL_SIZE, CHANNELS_PER_BLOCK))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUTS_PER_BLOCK, CHANNELS_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel elements
    for k in range(0, kernel_size):
        # Calculate input position
        input_pos = output_pos * stride + k * dilation - padding
        
        # Load input data
        input_data = tl.zeros((OUTPUTS_PER_BLOCK, CHANNELS_PER_BLOCK), dtype=tl.float32)
        
        # Check bounds for input
        valid_mask = (input_pos >= 0) & (input_pos < input_length)
        
        # Load weights
        weight_data = tl.load(weight_ptr + channel_id * kernel_size + k, mask=valid_mask, other=0.0)
        
        # Load input data
        if valid_mask:
            # Load input chunk
            input_chunk = tl.load(input_ptr + batch_id * in_channels * input_length + 
                                channel_id * input_length + input_pos, 
                                mask=valid_mask, other=0.0)
            input_data = input_chunk
            
        # Accumulate
        acc += input_data * weight_data
    
    # Apply bias if present
    if bias_ptr != 0:
        bias_data = tl.load(bias_ptr + channel_id, mask=True, other=0.0)
        acc += bias_data
    
    # Write output
    output_offset = batch_id * out_channels * output_length + channel_id * output_length + output_pos
    tl.store(output_ptr + output_offset, acc, mask=output_pos < output_length)

def triton_conv1d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Handle grouping
    if groups > 1:
        # For grouped convolution, implement as separate convolutions
        raise NotImplementedError("Grouped convolution not implemented")
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 8
    OUTPUTS_PER_BLOCK = 8
    
    # Grid dimensions
    batch_blocks = batch_size
    channel_blocks = (out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK
    output_blocks = (output_length + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK
    
    grid = (batch_blocks, channel_blocks, output_blocks)
    
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
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUTS_PER_BLOCK=OUTPUTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation using Triton optimizations.
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        """
        return triton_conv1d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )