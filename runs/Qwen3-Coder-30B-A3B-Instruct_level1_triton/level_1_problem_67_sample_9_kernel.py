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
    padding,
    dilation,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_BLOCK_SIZE: tl.constexpr,
    CHANNELS_BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_pos_id = tl.program_id(2)
    
    # Calculate global output position
    output_pos = out_pos_id * BLOCK_SIZE
    
    # Shared memory for input window
    input_window = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_BLOCK_SIZE, CHANNELS_BLOCK_SIZE, kernel_size))
    
    # Process each group
    for g in range(groups):
        # Calculate channel indices for this group
        ch_start = g * (in_channels // groups)
        ch_end = (g + 1) * (in_channels // groups)
        
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Load bias if available
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + out_ch_id)
            acc = bias_val
        
        # Process kernel elements
        for k in range(kernel_size):
            # Calculate input position
            input_pos = output_pos * stride - padding + k * dilation
            
            # Check bounds
            valid_input = (input_pos >= 0) & (input_pos < input_length)
            
            # Load input values
            input_vals = tl.zeros((CHANNELS_BLOCK_SIZE,), dtype=tl.float32)
            if valid_input:
                for c in range(ch_start, ch_end):
                    input_idx = batch_id * in_channels * input_length + c * input_length + input_pos
                    input_vals[c - ch_start] = tl.load(input_ptr + input_idx, mask=True)
            
            # Load weight values
            weight_vals = tl.zeros((CHANNELS_BLOCK_SIZE,), dtype=tl.float32)
            for c in range(ch_start, ch_end):
                weight_idx = out_ch_id * in_channels * kernel_size + c * kernel_size + k
                weight_vals[c - ch_start] = tl.load(weight_ptr + weight_idx, mask=True)
            
            # Compute dot product
            for c in range(ch_start, ch_end):
                acc += input_vals[c - ch_start] * weight_vals[c - ch_start]
        
        # Store output
        if output_pos < output_length:
            output_idx = batch_id * out_channels * output_length + out_ch_id * output_length + output_pos
            tl.store(output_ptr + output_idx, acc)

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
    if bias is not None:
        bias = bias.contiguous()
    
    # Define grid dimensions
    grid = (
        batch_size,  # Batch dimension
        out_channels,  # Output channels
        (output_length + 127) // 128  # Output positions (with block size 128)
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
        BLOCK_SIZE=128,
        GROUPS_BLOCK_SIZE=1,
        CHANNELS_BLOCK_SIZE=32
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution using Triton kernel.
        """
        return triton_conv1d(x, self.weight, self.bias, 
                           stride=self.stride, 
                           padding=self.padding, 
                           dilation=self.dilation, 
                           groups=self.groups)