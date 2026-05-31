import torch
import torch.nn as nn
import torch.nn.functional as F
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
    GROUP_SIZE: tl.constexpr,
    USE_BIAS: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which output row this program handles
    output_row = pid // GROUP_SIZE
    group_id = pid % GROUP_SIZE
    
    if output_row >= output_length:
        return
        
    # Shared memory for weight caching
    shared_weight = tl.shared_ptr(weight_ptr, shape=(in_channels, out_channels, kernel_size), dtype=tl.float32)
    
    # Each program processes one output position
    for out_pos in range(output_row, output_length, GROUP_SIZE):
        # Initialize accumulator
        acc = tl.zeros((out_channels,), dtype=tl.float32)
        
        # Process each input channel and kernel position
        for k in range(kernel_size):
            # Calculate input position
            input_pos = out_pos - padding + k * dilation
            
            # Check bounds
            if input_pos >= 0 and input_pos < input_length:
                # Load input data for this position
                input_data = tl.load(input_ptr + 
                                   tl.arange(0, in_channels) + 
                                   input_pos * in_channels + 
                                   tl.arange(0, batch_size)[:, None] * input_length * in_channels,
                                   mask=(tl.arange(0, in_channels) < in_channels),
                                   other=0.0)
                
                # Load weight data
                weight_data = tl.load(weight_ptr + 
                                    tl.arange(0, in_channels)[:, None] + 
                                    tl.arange(0, out_channels)[None, :] + 
                                    k * in_channels * out_channels,
                                    mask=(tl.arange(0, in_channels)[:, None] < in_channels) &
                                          (tl.arange(0, out_channels)[None, :] < out_channels),
                                    other=0.0)
                
                # Accumulate
                acc += tl.sum(input_data[:, None] * weight_data, axis=0)
        
        # Add bias if needed
        if USE_BIAS:
            bias_data = tl.load(bias_ptr + tl.arange(0, out_channels), mask=(tl.arange(0, out_channels) < out_channels))
            acc += bias_data
        
        # Store output
        tl.store(output_ptr + 
                tl.arange(0, out_channels) + 
                out_pos * out_channels + 
                tl.arange(0, batch_size)[:, None] * output_length * out_channels,
                acc,
                mask=(tl.arange(0, out_channels) < out_channels))

def triton_conv1d_transpose(input_tensor, weight, bias, stride, padding, dilation):
    """
    Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256
    GROUP_SIZE = 8
    
    # Grid configuration
    num_programs = output_length * GROUP_SIZE
    grid = lambda meta: (num_programs,)
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
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
        BLOCK_SIZE,
        GROUP_SIZE,
        USE_BIAS=(bias is not None)
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version using Triton kernels for ConvTranspose1d operations
    """
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
        Performs the transposed 1D convolution using Triton kernel
        """
        return triton_conv1d_transpose(x, self.weight, self.bias, self.stride, self.padding, self.dilation)