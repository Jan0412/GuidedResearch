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
    # Get program IDs
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate global indices
    batch_offset = batch_idx * in_channels * input_length
    channel_offset = channel_idx * input_length
    output_offset = output_idx * out_channels * output_length
    
    # Shared memory for input window
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(CHANNELS_PER_BLOCK, kernel_size * dilation + 1))
    
    # Process multiple channels per block
    for c in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load input window into shared memory
        for i in range(kernel_size):
            idx = output_idx * stride + i * dilation
            if idx >= 0 and idx < input_length:
                for ch in range(CHANNELS_PER_BLOCK):
                    if c + ch < in_channels:
                        shared_input[ch, i] = tl.load(input_ptr + batch_offset + (c + ch) * input_length + idx, mask=(idx < input_length))
                    else:
                        shared_input[ch, i] = 0.0
        
        # Compute convolution for this output position
        for oc in range(OUTPUTS_PER_BLOCK):
            if c + oc < out_channels:
                acc = 0.0
                for k in range(kernel_size):
                    for ch in range(CHANNELS_PER_BLOCK):
                        if c + ch < in_channels:
                            w_val = tl.load(weight_ptr + (c + ch) * out_channels + (c + oc), mask=(c + ch < in_channels))
                            acc += w_val * shared_input[ch, k]
                
                # Add bias if present
                if bias_ptr is not None:
                    bias_val = tl.load(bias_ptr + (c + oc), mask=(c + oc < out_channels))
                    acc += bias_val
                
                # Store result
                tl.store(output_ptr + output_offset + (c + oc) * output_length + output_idx, acc)

def triton_conv1d(input_tensor, weight, bias, stride, dilation):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * 0 - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 8
    OUTPUTS_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,           # batch dimension
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,  # channel dimension
        output_length         # output position dimension
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
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.
    Optimized with Triton kernels.
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
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)