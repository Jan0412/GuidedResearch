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
    GROUPS: tl.constexpr
):
    # Get program ID
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Calculate group info
    channels_per_group = out_channels // groups
    group_id = channel_id // channels_per_group
    
    # Each program processes one output position
    output_pos = tl.program_id(2)
    
    if output_pos >= output_length:
        return
        
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input positions and kernel weights
    for k in range(kernel_size):
        # Calculate input position
        input_pos = output_pos - padding + k * stride
        
        # Check bounds
        if input_pos >= 0 and input_pos < input_length:
            # Calculate pointers
            input_offset = batch_id * input_stride_0 + channel_id * input_stride_1 + input_pos * input_stride_2
            weight_offset = channel_id * weight_stride_0 + group_id * weight_stride_1 + k * weight_stride_2
            
            # Load input value
            input_val = tl.load(input_ptr + input_offset, mask=(input_pos < input_length))
            
            # Load weight value
            weight_val = tl.load(weight_ptr + weight_offset)
            
            # Accumulate
            acc += input_val * weight_val
    
    # Write output
    output_offset = batch_id * output_stride_0 + channel_id * output_stride_1 + output_pos * output_stride_2
    tl.store(output_ptr + output_offset, acc)

def triton_conv1d_transpose(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
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
    output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=input_tensor.device, dtype=torch.float32)
    
    # Set up grid
    grid = (
        batch_size,
        out_channels,
        output_length
    )
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        input_tensor,
        weight,
        output,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
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
        GROUPS=groups
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
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
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
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )

# Keep original Model for comparison
class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation.

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
        super(Model, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d_transpose(x)