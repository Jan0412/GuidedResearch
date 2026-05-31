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
    input_window = tl.shared_memory(dtype=tl.float32, shape=(CHANNELS_PER_BLOCK, kernel_size))
    
    # Process each output position
    for out_pos in range(output_idx * OUTPUTS_PER_BLOCK, min((output_idx + 1) * OUTPUTS_PER_BLOCK, output_length)):
        # Initialize accumulator
        acc = tl.zeros((OUT_CHANNELS_PER_BLOCK,), dtype=tl.float32)
        
        # Load bias if available
        if bias_ptr is not None:
            bias_offset = channel_idx * out_channels
            acc = tl.load(bias_ptr + bias_offset, mask=channel_idx < out_channels)
        
        # Load input window for this output position
        input_base_pos = out_pos * stride
        
        # Process kernel elements
        for k in range(kernel_size):
            # Calculate input position with dilation
            input_pos = input_base_pos + k * dilation
            
            # Check bounds
            valid_input = (input_pos >= 0) & (input_pos < input_length)
            
            # Load input data for this kernel position
            for c in range(CHANNELS_PER_BLOCK):
                if c + channel_idx < in_channels:
                    input_val = tl.load(input_ptr + batch_offset + (c + channel_idx) * input_length + input_pos, 
                                      mask=valid_input, other=0.0)
                    weight_val = tl.load(weight_ptr + channel_idx * out_channels * kernel_size + 
                                       c * out_channels * kernel_size + k * out_channels + out_pos % out_channels)
                    acc += input_val * weight_val
        
        # Store output
        output_pos_global = batch_idx * out_channels * output_length + out_pos * out_channels + channel_idx
        tl.store(output_ptr + output_pos_global, acc, mask=channel_idx < out_channels)


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
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256
    CHANNELS_PER_BLOCK = 16
    OUTPUTS_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,  # Batch dimension
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,  # Channel dimension
        (output_length + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK  # Output position dimension
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
        dilation,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUTS_PER_BLOCK=OUTPUTS_PER_BLOCK
    )
    
    return output


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, 
    potentially dilated and strided, optimized with Triton kernels.
    """
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
            
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel optimization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)