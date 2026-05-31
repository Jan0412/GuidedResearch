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
    GROUPS_BLOCK_SIZE: tl.constexpr,
    CHANNELS_BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    output_pos_id = tl.program_id(2)
    
    # Shared memory for weight tiles
    shared_weight = tl.shared_block([CHANNELS_BLOCK_SIZE, KERNEL_SIZE])
    
    # Calculate output position and input positions
    output_pos = output_pos_id * stride - padding
    
    # Each thread processes one output element
    if output_pos >= 0 and output_pos < input_length - (kernel_size - 1) * dilation:
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Process each group
        for g in range(groups):
            # Calculate channel indices
            in_channel_start = g * (in_channels // groups)
            out_channel_start = g * (out_channels // groups)
            
            # Check if this thread should process this group
            if out_channel_id >= out_channel_start and out_channel_id < out_channel_start + (out_channels // groups):
                # Load weight tile
                weight_offset = (out_channel_id - out_channel_start) * (in_channels // groups) * kernel_size + \
                               (g * (in_channels // groups) * kernel_size)
                
                # Load input data for this group
                for k in range(kernel_size):
                    input_pos = output_pos + k * dilation
                    if input_pos >= 0 and input_pos < input_length:
                        for c in range(in_channels // groups):
                            if c < (in_channels // groups):
                                input_idx = batch_id * in_channels * input_length + \
                                          (in_channel_start + c) * input_length + input_pos
                                weight_idx = weight_offset + c * kernel_size + k
                                
                                input_val = tl.load(input_ptr + input_idx, mask=(input_pos < input_length))
                                weight_val = tl.load(weight_ptr + weight_idx, mask=(k < kernel_size))
                                acc += input_val * weight_val
                
                # Add bias if available
                if bias_ptr is not None:
                    bias_offset = out_channel_id
                    bias_val = tl.load(bias_ptr + bias_offset)
                    acc += bias_val
                
                # Store result
                output_idx = batch_id * out_channels * output_length + \
                           out_channel_id * output_length + output_pos_id
                tl.store(output_ptr + output_idx, acc)

def triton_conv1d(input_tensor, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    """
    Triton implementation of 1D convolution
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define block sizes
    BLOCK_SIZE = 128
    GROUPS_BLOCK_SIZE = 32
    CHANNELS_BLOCK_SIZE = 32
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_length
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
        padding,
        dilation,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS_BLOCK_SIZE=GROUPS_BLOCK_SIZE,
        CHANNELS_BLOCK_SIZE=CHANNELS_BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation using Triton kernels for optimization.
    """
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
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)