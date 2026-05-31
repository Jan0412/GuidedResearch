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
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    group_idx = tl.program_id(2)
    
    # Calculate output position
    output_pos = tl.program_id(3) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Check bounds
    mask = output_pos < output_length
    
    # Shared memory for input chunk
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over kernel positions
    for kernel_pos in range(kernel_size):
        # Calculate input position
        input_pos = output_pos * stride + kernel_pos - padding
        
        # Create mask for valid input positions
        input_mask = (input_pos >= 0) & (input_pos < input_length)
        
        # Load input data
        input_chunk = tl.load(input_ptr + 
                             batch_idx * in_channels * input_length +
                             group_idx * (in_channels // groups) * input_length +
                             input_pos, 
                             mask=input_mask & mask, other=0.0)
        
        # Load weight
        weight_val = tl.load(weight_ptr + 
                           out_channel_idx * groups * kernel_size * (in_channels // groups) +
                           group_idx * kernel_size * (in_channels // groups) +
                           kernel_pos * (in_channels // groups) +
                           tl.arange(0, BLOCK_SIZE) % (in_channels // groups),
                           mask=mask, other=0.0)
        
        # Accumulate
        acc += input_chunk * weight_val
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * out_channels * output_length +
             out_channel_idx * output_length +
             output_pos,
             acc, mask=mask)

def triton_conv1d_transpose(input_tensor, weight, bias=None, stride=1, padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_length = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - 1) * stride - 2 * padding + kernel_size
    
    # Prepare output tensor
    output = torch.zeros(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Launch kernel
    grid = (
        batch_size,  # batch dimension
        out_channels,  # output channel dimension
        groups,  # group dimension
        (output_length + 127) // 128  # output position dimension
    )
    
    BLOCK_SIZE = 128
    GROUP_SIZE = 1
    
    # For simplicity, using a basic approach without shared memory optimization
    # In practice, this would be more complex to achieve optimal performance
    conv1d_transpose_kernel[grid](
        input_tensor,
        weight,
        output,
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
        GROUP_SIZE=GROUP_SIZE
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation using Triton kernels for speedup.
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
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
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
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )