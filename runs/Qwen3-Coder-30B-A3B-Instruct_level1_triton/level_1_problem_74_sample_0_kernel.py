import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv1d_transpose_kernel(
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
    OUTPUT_LENGTH_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_pos_idx = tl.program_id(2)
    
    # Calculate output position
    output_pos_start = output_pos_idx * OUTPUT_LENGTH_PER_BLOCK
    output_pos_end = min(output_pos_start + OUTPUT_LENGTH_PER_BLOCK, output_length)
    
    # Shared memory for input chunk
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(CHANNELS_PER_BLOCK, OUTPUT_LENGTH_PER_BLOCK))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_LENGTH_PER_BLOCK,), dtype=tl.float32)
    
    # Loop over kernel positions
    for kernel_pos in range(kernel_size):
        # Calculate input position (considering stride and dilation)
        input_pos = output_pos_start + kernel_pos * dilation - padding
        
        # Check if we're within valid input range
        valid_input = (input_pos >= 0) & (input_pos < input_length)
        
        # Load weights for this kernel position
        weight_vals = tl.load(weight_ptr + 
                             channel_idx * in_channels * kernel_size + 
                             kernel_pos * in_channels + 
                             tl.arange(0, CHANNELS_PER_BLOCK), 
                             mask=(tl.arange(0, CHANNELS_PER_BLOCK) < in_channels) & (channel_idx < out_channels))
        
        # Load input data
        input_vals = tl.load(input_ptr + 
                            batch_idx * in_channels * input_length + 
                            tl.arange(0, CHANNELS_PER_BLOCK) * input_length + 
                            input_pos, 
                            mask=valid_input & (tl.arange(0, CHANNELS_PER_BLOCK) < in_channels))
        
        # Accumulate
        acc += tl.sum(weight_vals[:, None] * input_vals[None, :], axis=0)
    
    # Apply bias if available
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel_idx, mask=channel_idx < out_channels)
        acc += bias_val
    
    # Write output
    tl.store(output_ptr + 
             batch_idx * out_channels * output_length + 
             channel_idx * output_length + 
             tl.arange(0, OUTPUT_LENGTH_PER_BLOCK),
             acc,
             mask=(tl.arange(0, OUTPUT_LENGTH_PER_BLOCK) < output_length - output_pos_start))

class Conv1dTransposeTriton(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(Conv1dTransposeTriton, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        # x shape: (batch_size, in_channels, input_length)
        batch_size, _, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.kernel_size * self.dilation
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=torch.float32)
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Handle bias
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        
        # Configure kernel launch parameters
        BLOCK_SIZE = 256
        CHANNELS_PER_BLOCK = 32
        OUTPUT_LENGTH_PER_BLOCK = 64
        
        # Grid dimensions
        grid = (
            batch_size,
            (self.out_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
            (output_length + OUTPUT_LENGTH_PER_BLOCK - 1) // OUTPUT_LENGTH_PER_BLOCK
        )
        
        # Launch kernel
        conv1d_transpose_kernel[grid](
            x,
            weight,
            output,
            bias_ptr,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_length,
            output_length,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            BLOCK_SIZE=BLOCK_SIZE,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
            OUTPUT_LENGTH_PER_BLOCK=OUTPUT_LENGTH_PER_BLOCK
        )
        
        return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = Conv1dTransposeTriton(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d_transpose(x)