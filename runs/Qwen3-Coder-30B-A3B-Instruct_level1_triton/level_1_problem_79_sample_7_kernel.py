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
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Each program processes one output row (batch, channel, position)
    num_programs = batch_size * out_channels * output_length
    program_id = pid
    
    # Compute which batch, channel, and output position this program handles
    output_pos = program_id % output_length
    channel = (program_id // output_length) % out_channels
    batch = (program_id // (output_length * out_channels)) % batch_size
    
    # Compute input position
    input_pos = (output_pos - padding) // stride
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel elements
    for k in range(kernel_size):
        # Compute kernel index
        kernel_idx = k
        
        # Compute input index
        input_idx = input_pos - (kernel_size - 1 - k) * dilation
        
        # Check bounds
        if input_idx >= 0 and input_idx < input_length:
            # Load input value
            input_val = tl.load(input_ptr + 
                               batch * in_channels * input_length +
                               channel * input_length +
                               input_idx,
                               mask=(input_idx < input_length),
                               other=0.0)
            
            # Load weight value
            weight_val = tl.load(weight_ptr + 
                                channel * out_channels * kernel_size +
                                (channel * kernel_size + kernel_idx),
                                mask=True,
                                other=0.0)
            
            # Accumulate
            acc += input_val * weight_val
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel, mask=True, other=0.0)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + 
             batch * out_channels * output_length +
             channel * output_length +
             output_pos,
             acc,
             mask=(output_pos < output_length))

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
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        batch_size, _, input_length = x.shape
        
        # Calculate output length
        output_length = (input_length - 1) * self.stride - 2 * self.padding + (self.kernel_size - 1) * self.dilation + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_length, device=x.device, dtype=torch.float32)
        
        # Ensure tensors are contiguous and on correct device
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Handle bias
        bias_ptr = self.bias_param.data if self.bias else None
        
        # Launch kernel
        grid_size = batch_size * self.out_channels * output_length
        BLOCK_SIZE = 128
        GROUP_SIZE_M = 8
        
        # Create grid
        grid = lambda meta: (grid_size,)
        
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
            GROUP_SIZE_M=GROUP_SIZE_M
        )
        
        return output