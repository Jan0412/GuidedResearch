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
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(CHANNELS_PER_BLOCK, kernel_size))
    
    # Calculate output position
    out_pos = output_idx * stride
    
    # Each block processes one output position
    if out_pos >= output_length:
        return
        
    # Process multiple channels per block
    for c in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weights for this channel and output
        weight_offsets = c * out_channels * kernel_size + channel_idx * kernel_size
        weights = tl.load(weight_ptr + weight_offsets + tl.arange(0, kernel_size), mask=(tl.arange(0, kernel_size) < kernel_size))
        
        # Load input window
        input_offsets = batch_idx * in_channels * input_length + c * input_length + out_pos * stride
        for k in range(kernel_size):
            input_pos = input_offsets + k * dilation
            shared_input[tl.arange(0, CHANNELS_PER_BLOCK), k] = tl.load(input_ptr + input_pos + tl.arange(0, CHANNELS_PER_BLOCK), mask=(tl.arange(0, CHANNELS_PER_BLOCK) < in_channels - c))
        
        # Compute convolution for current channel
        acc = 0.0
        for k in range(kernel_size):
            for ch in range(CHANNELS_PER_BLOCK):
                if c + ch < in_channels:
                    acc += shared_input[ch, k] * weights[k]
        
        # Store output
        output_offset = batch_idx * out_channels * output_length + channel_idx * output_length + output_idx
        tl.store(output_ptr + output_offset, acc)

def triton_conv1d(input_tensor, weight, bias, stride, dilation, padding=0):
    """
    Custom Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
    
    # Initialize output tensor
    output = torch.zeros(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define block sizes
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 32
    OUTPUTS_PER_BLOCK = 16
    
    # Grid configuration
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        (output_length + OUTPUTS_PER_BLOCK - 1) // OUTPUTS_PER_BLOCK
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
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUTS_PER_BLOCK=OUTPUTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, 
    potentially dilated and strided, using custom Triton kernels for optimization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights and bias
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
        Performs the 1D convolution using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Use custom Triton kernel for convolution
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)