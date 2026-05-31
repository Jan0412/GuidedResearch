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
    input_size,
    output_size,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    OUTPUT_SIZE_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    output_id = tl.program_id(2)
    
    # Calculate output position
    output_start = output_id * OUTPUT_SIZE_PER_BLOCK
    output_end = tl.minimum(output_start + OUTPUT_SIZE_PER_BLOCK, output_size)
    
    # Shared memory for input cache
    input_cache = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Loop over kernel positions
    for k in range(kernel_size):
        # Calculate input position
        input_pos = output_start + k * stride - padding
        
        # Load input data
        input_offset = batch_id * in_channels * input_size + group_id * CHANNELS_PER_GROUP * input_size
        input_ptr_local = input_ptr + input_offset
        
        # Load kernel weights
        weight_offset = group_id * CHANNELS_PER_GROUP * out_channels * kernel_size
        weight_ptr_local = weight_ptr + weight_offset
        
        # Process output positions
        for out_pos in range(output_start, output_end):
            if out_pos >= 0 and out_pos < input_size:
                # Load input value
                input_val = tl.load(input_ptr_local + out_pos, mask=(out_pos < input_size))
                
                # Compute output for each channel in group
                for c in range(CHANNELS_PER_GROUP):
                    if c < CHANNELS_PER_GROUP:
                        # Compute weight index
                        weight_idx = c * kernel_size + k
                        weight_val = tl.load(weight_ptr_local + weight_idx, mask=(weight_idx < CHANNELS_PER_GROUP * kernel_size))
                        
                        # Accumulate output
                        output_idx = batch_id * out_channels * output_size + group_id * CHANNELS_PER_GROUP * output_size + c * output_size + out_pos
                        tl.atomic_add(output_ptr + output_idx, input_val * weight_val)

def triton_conv1d_transpose(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_size = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output size
    output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_size, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Constants
    BLOCK_SIZE = 256
    GROUPS = groups
    CHANNELS_PER_GROUP = in_channels // groups
    
    # Grid configuration
    grid = (
        batch_size,  # batch dimension
        groups,      # group dimension
        (output_size + 255) // 256  # output position dimension
    )
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        input_tensor,
        weight,
        output,
        input_size,
        output_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=GROUPS,
        CHANNELS_PER_GROUP=CHANNELS_PER_GROUP,
        OUTPUT_SIZE_PER_BLOCK=256
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation using Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
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
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )

# Note: This implementation provides a basic Triton kernel for ConvTranspose1d.
# For production use, further optimizations including better memory access patterns,
# shared memory usage, and fused operations would be beneficial.