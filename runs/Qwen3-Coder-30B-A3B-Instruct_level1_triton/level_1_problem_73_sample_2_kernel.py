import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_depth,
    input_height,
    input_width,
    output_depth,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    
    # Calculate which group this output channel belongs to
    group_id = out_ch_idx // group_size
    
    # Shared memory for output accumulation
    output_block = tl.zeros((OUTPUT_BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for k_d in range(kernel_size):
        for k_h in range(kernel_size):
            for k_w in range(kernel_size):
                # Calculate input position
                input_d = out_d_idx * stride + k_d - padding
                input_h = out_d_idx * stride + k_h - padding
                input_w = out_d_idx * stride + k_w - padding
                
                # Check bounds
                if input_d >= 0 and input_d < input_depth and \
                   input_h >= 0 and input_h < input_height and \
                   input_w >= 0 and input_w < input_width:
                    
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * in_channels * input_depth * input_height * input_width +
                                       group_id * group_size * input_depth * input_height * input_width +
                                       input_d * input_height * input_width +
                                       input_h * input_width +
                                       input_w)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr +
                                        out_ch_idx * kernel_size * kernel_size * kernel_size * in_channels // groups +
                                        k_d * kernel_size * kernel_size * in_channels // groups +
                                        k_h * kernel_size * in_channels // groups +
                                        k_w * in_channels // groups +
                                        (out_ch_idx % group_size))
                    
                    # Accumulate
                    output_block += input_val * weight_val
    
    # Store output
    output_offset = batch_idx * out_channels * output_depth * output_height * output_width + \
                   out_ch_idx * output_depth * output_height * output_width + \
                   out_d_idx * output_height * output_width
    
    tl.store(output_ptr + output_offset, output_block)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=1, padding=0, groups=1):
    """
    Custom Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride - 2 * padding + kernel_depth
    output_height = (input_height - 1) * stride - 2 * padding + kernel_height
    output_width = (input_width - 1) * stride - 2 * padding + kernel_width
    
    # Prepare output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Set up launch parameters
    BLOCK_SIZE = 1024
    OUTPUT_BLOCK_SIZE = 32
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_depth
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_depth,
        stride,
        padding,
        groups,
        out_channels // groups,
        BLOCK_SIZE,
        OUTPUT_BLOCK_SIZE
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
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
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, groups={self.groups}, bias={self.bias is not None}'