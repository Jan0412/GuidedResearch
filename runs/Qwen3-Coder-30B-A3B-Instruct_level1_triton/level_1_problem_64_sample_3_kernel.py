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
    input_stride_0, input_stride_1, input_stride_2,
    weight_stride_0, weight_stride_1, weight_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_pos_id = tl.program_id(2)
    
    # Shared memory for group processing
    shared_weight = tl.shared_memory(shape=(GROUPS_BLOCK_SIZE, kernel_size), dtype=tl.float32)
    
    # Calculate global output position
    output_pos = out_pos_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process groups
    for g in range(groups):
        # Calculate group-specific indices
        group_in_ch = g * (in_channels // groups)
        group_out_ch = g * (out_channels // groups)
        
        # Load weight for this group and kernel position
        if out_ch_id < (out_channels // groups):
            for k in range(kernel_size):
                w_idx = group_out_ch + out_ch_id
                k_idx = k
                shared_weight[g, k] = tl.load(weight_ptr + 
                                            w_idx * weight_stride_0 + 
                                            k_idx * weight_stride_2)
        
        tl.sync()
        
        # Compute convolution for current group
        for k in range(kernel_size):
            # Calculate input positions
            input_pos = output_pos * stride + k - padding
            
            # Create mask for valid positions
            valid_mask = (input_pos >= 0) & (input_pos < input_length)
            
            # Load input values
            input_vals = tl.load(input_ptr + 
                               batch_id * input_stride_0 + 
                               (group_in_ch + out_ch_id % (in_channels // groups)) * input_stride_1 +
                               input_pos, 
                               mask=valid_mask, other=0.0)
            
            # Load weight for current kernel position
            weight_val = shared_weight[g, k]
            
            # Accumulate
            acc += input_vals * weight_val
        
        tl.sync()
    
    # Write output
    if out_ch_id < out_channels and out_pos_id * BLOCK_SIZE < output_length:
        output_vals = acc
        output_mask = (output_pos < output_length)
        tl.store(output_ptr + 
                batch_id * output_stride_0 + 
                out_ch_id * output_stride_1 + 
                output_pos, 
                output_vals, 
                mask=output_mask)

def triton_conv1d_transpose(input_tensor, weight, bias=None, stride=1, padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + kernel_size
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 128
    GROUPS_BLOCK_SIZE = min(groups, 32)
    
    # Grid configuration
    grid = (
        batch_size,           # batch dimension
        out_channels,         # output channel dimension  
        (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE  # output position dimension
    )
    
    # Define strides for memory access
    input_stride_0 = input_tensor.stride(0)
    input_stride_1 = input_tensor.stride(1)
    input_stride_2 = input_tensor.stride(2)
    
    weight_stride_0 = weight.stride(0)
    weight_stride_1 = weight.stride(1)
    weight_stride_2 = weight.stride(2)
    
    output_stride_0 = output.stride(0)
    output_stride_1 = output.stride(1)
    output_stride_2 = output.stride(2)
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        input_tensor,
        weight,
        output,
        input_stride_0, input_stride_1, input_stride_2,
        weight_stride_0, weight_stride_1, weight_stride_2,
        output_stride_0, output_stride_1, output_stride_2,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS_BLOCK_SIZE=GROUPS_BLOCK_SIZE
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation using Triton kernels for optimization.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Use our Triton kernel implementation
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, stride={self.stride}, '
            f'padding={self.padding}, output_padding={self.output_padding}, '
            f'groups={self.groups}, bias={self.bias is not None}'
        )