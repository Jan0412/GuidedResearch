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
    dilation,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output element
    # Calculate which output element this program handles
    output_idx = pid
    
    if output_idx >= batch_size * out_channels * output_length:
        return
        
    # Calculate batch, channel, and position indices
    batch_idx = output_idx // (out_channels * output_length)
    remaining = output_idx % (out_channels * output_length)
    channel_idx = remaining // output_length
    pos_idx = remaining % output_length
    
    # Calculate input start position for this output position
    input_start_pos = pos_idx * stride
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Perform convolution
    for k in range(kernel_size):
        # Calculate input position with dilation
        input_pos = input_start_pos + k * dilation
        
        # Check bounds
        if input_pos >= 0 and input_pos < input_length:
            # Load input value
            input_val = tl.load(input_ptr + 
                               batch_idx * (in_channels * input_length) +
                               channel_idx * input_length +
                               input_pos)
            
            # Load weight value
            weight_val = tl.load(weight_ptr + 
                                channel_idx * (out_channels * kernel_size) +
                                channel_idx * kernel_size +
                                k)
            
            acc += input_val * weight_val
    
    # Add bias if enabled
    if bias_enabled:
        bias_val = tl.load(bias_ptr + channel_idx)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + output_idx, acc[0])

def triton_conv1d(input_tensor, weight, bias, stride, dilation):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * 0 - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, dtype=torch.float32, device=input_tensor.device)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Handle bias
    bias_enabled = bias is not None
    if bias_enabled:
        bias = bias.contiguous()
    
    # Launch kernel
    total_elements = batch_size * out_channels * output_length
    BLOCK_SIZE = 1024
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
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
        bias_enabled,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=32
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        """
        # Use our Triton implementation instead of PyTorch's native conv1d
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.dilation)