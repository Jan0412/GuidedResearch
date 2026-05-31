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
    channel_group = tl.program_id(1)
    output_pos_group = tl.program_id(2)
    
    # Calculate starting positions
    start_channel = channel_group * CHANNELS_PER_BLOCK
    start_output_pos = output_pos_group * OUTPUT_LENGTH_PER_BLOCK
    
    # Shared memory for input window
    input_window = tl.shared_memory(dtype=tl.float32, shape=(KERNEL_SIZE, CHANNELS_PER_BLOCK))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_LENGTH_PER_BLOCK, CHANNELS_PER_BLOCK), dtype=tl.float32)
    
    # Loop over kernel positions
    for k in range(0, kernel_size):
        # Calculate input position (considering dilation and stride)
        input_pos = start_output_pos + k * dilation - padding
        
        # Check bounds for input
        valid_input = (input_pos >= 0) & (input_pos < input_length)
        
        # Load input data
        if valid_input:
            # Load from input tensor
            input_data = tl.load(input_ptr + 
                               batch_idx * in_channels * input_length +
                               start_channel * input_length +
                               input_pos * in_channels,
                               mask=valid_input & (tl.arange(0, CHANNELS_PER_BLOCK) < in_channels - start_channel),
                               other=0.0)
            
            # Load weights for this kernel position
            weight_data = tl.load(weight_ptr + 
                                start_channel * kernel_size + 
                                k * in_channels,
                                mask=tl.arange(0, CHANNELS_PER_BLOCK) < in_channels - start_channel,
                                other=0.0)
            
            # Accumulate
            acc += tl.expand_dims(input_data, 0) * tl.expand_dims(weight_data, 1)
    
    # Add bias if available
    if bias_ptr is not None:
        bias_data = tl.load(bias_ptr + start_channel,
                           mask=tl.arange(0, CHANNELS_PER_BLOCK) < in_channels - start_channel,
                           other=0.0)
        acc += tl.expand_dims(bias_data, 0)
    
    # Store results
    tl.store(output_ptr + 
             batch_idx * out_channels * output_length +
             start_channel * output_length +
             start_output_pos,
             acc,
             mask=tl.arange(0, OUTPUT_LENGTH_PER_BLOCK) < output_length - start_output_pos &
                  (tl.arange(0, CHANNELS_PER_BLOCK) < in_channels - start_channel))

def triton_conv1d_transpose(input_tensor, weight, bias, stride, padding, dilation):
    """
    Triton implementation of ConvTranspose1d using custom kernel
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Define block sizes
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 32
    OUTPUT_LENGTH_PER_BLOCK = 64
    
    # Grid dimensions
    grid = (
        batch_size,  # Batch dimension
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,  # Channel groups
        (output_length + OUTPUT_LENGTH_PER_BLOCK - 1) // OUTPUT_LENGTH_PER_BLOCK  # Output position groups
    )
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
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
        dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_LENGTH_PER_BLOCK=OUTPUT_LENGTH_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
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
        Performs the transposed 1D convolution using Triton kernel.
        """
        return triton_conv1d_transpose(x, self.weight, self.bias, self.stride, self.padding, self.dilation)