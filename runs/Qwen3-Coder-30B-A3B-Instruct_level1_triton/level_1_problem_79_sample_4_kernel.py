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
    GROUP_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which output channel this program handles
    output_channel_id = pid // (batch_size * output_length)
    program_id = pid % (batch_size * output_length)
    
    # Get batch and position
    batch_id = program_id // output_length
    pos_id = program_id % output_length
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Handle bias if present
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + output_channel_id, mask=output_channel_id < out_channels)
    
    # Loop over input channels and kernel positions
    for ic in range(in_channels):
        for ki in range(kernel_size):
            # Calculate input position
            input_pos = pos_id * stride - padding + ki * dilation
            
            # Check bounds
            if input_pos >= 0 and input_pos < input_length:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_id * in_channels * input_length +
                                  ic * input_length +
                                  input_pos,
                                  mask=input_pos < input_length)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   output_channel_id * in_channels * kernel_size +
                                   ic * kernel_size +
                                   ki,
                                   mask=ki < kernel_size)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + 
             batch_id * out_channels * output_length +
             output_channel_id * output_length +
             pos_id,
             acc,
             mask=pos_id < output_length)

def triton_conv1d_transpose(input_tensor, weight, bias, stride, padding, dilation):
    """
    Triton implementation of Conv1dTranspose
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + (kernel_size - 1) * dilation + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Grid configuration
    total_elements = batch_size * out_channels * output_length
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv1d_transpose_kernel[grid_size](
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
        GROUP_SIZE=8
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
        Performs the transposed 1D convolution using Triton kernel.
        """
        return triton_conv1d_transpose(x, self.weight, self.bias, self.stride, self.padding, self.dilation)