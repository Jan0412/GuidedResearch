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
    padding,
    dilation,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    OUTPUT_CHANNELS_PER_GROUP: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_ch_idx = tl.program_id(2)
    
    # Calculate global output channel index
    global_out_ch = group_idx * OUTPUT_CHANNELS_PER_GROUP + (out_ch_idx % OUTPUT_CHANNELS_PER_GROUP)
    
    # Shared memory for input window
    input_window = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for ch in range(CHANNELS_PER_GROUP):
        # Calculate input channel index within group
        global_in_ch = group_idx * CHANNELS_PER_GROUP + ch
        
        # For each kernel position
        for k in range(kernel_size):
            # Calculate input position
            input_pos = batch_idx * input_length + global_in_ch * input_length
            
            # Load input window into shared memory
            for i in range(BLOCK_SIZE):
                if i < input_length:
                    input_window[i] = tl.load(input_ptr + input_pos + i, mask=(i < input_length))
            
            # Compute convolution for this kernel position
            for i in range(output_length):
                # Calculate kernel weight index
                weight_idx = global_out_ch * in_channels * kernel_size + global_in_ch * kernel_size + k
                
                # Calculate input index for current output position
                input_idx = i * stride - padding + k * dilation
                
                # Check if input position is valid
                if input_idx >= 0 and input_idx < input_length:
                    # Load input value and weight
                    input_val = tl.load(input_ptr + input_pos + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + global_out_ch, mask=True)
        acc += bias_val
    
    # Store result
    output_idx = batch_idx * out_channels * output_length + global_out_ch * output_length
    for i in range(output_length):
        tl.store(output_ptr + output_idx + i, acc, mask=True)


def triton_conv1d(input_tensor, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Handle bias
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Set up launch parameters
    BLOCK_SIZE = 128
    GROUPS = groups
    CHANNELS_PER_GROUP = in_channels // groups
    OUTPUT_CHANNELS_PER_GROUP = out_channels // groups
    
    # Grid dimensions
    grid = (
        batch_size,  # batch dimension
        groups,      # groups dimension
        OUTPUT_CHANNELS_PER_GROUP  # output channels per group
    )
    
    # Launch kernel
    conv1d_kernel[grid](
        input_tensor,
        weight,
        output,
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
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=GROUPS,
        CHANNELS_PER_GROUP=CHANNELS_PER_GROUP,
        OUTPUT_CHANNELS_PER_GROUP=OUTPUT_CHANNELS_PER_GROUP
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
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
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)