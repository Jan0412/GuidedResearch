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
    input_size,
    output_size,
    weight_size,
    batch_size,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_pos_idx = tl.program_id(2)
    
    # Calculate group info
    ch_per_group = out_channels // groups
    group_idx = out_ch_idx // ch_per_group
    
    # Shared memory for weight and input
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(GROUP_SIZE, kernel_size))
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(GROUP_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel elements
    for k in range(0, kernel_size):
        # Calculate input position
        input_pos = out_pos_idx * stride + k - padding
        
        # Check bounds
        if input_pos >= 0 and input_pos < input_size:
            # Load weight
            weight_val = tl.load(weight_ptr + 
                               (group_idx * ch_per_group + out_ch_idx % ch_per_group) * kernel_size + k)
            
            # Load input
            input_val = tl.load(input_ptr + 
                              batch_idx * in_channels * input_size +
                              (group_idx * ch_per_group + out_ch_idx % ch_per_group) * input_size +
                              input_pos)
            
            acc += weight_val * input_val
    
    # Store result
    if out_pos_idx < output_size:
        tl.store(output_ptr + 
                batch_idx * out_channels * output_size +
                out_ch_idx * output_size + 
                out_pos_idx, 
                acc[0])

def triton_conv1d_transpose(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose1d
    """
    batch_size, in_channels, input_size = input_tensor.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output size
    output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_size, device=input_tensor.device, dtype=torch.float32)
    
    # Create grid
    grid = (
        batch_size,
        out_channels,
        output_size
    )
    
    # Define block sizes
    BLOCK_SIZE = 128
    GROUP_SIZE = 32
    
    # Launch kernel
    conv1d_transpose_kernel[grid](
        input_tensor,
        weight,
        output,
        input_size,
        output_size,
        kernel_size,
        batch_size,
        in_channels,
        out_channels,
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
    Performs a transposed 1D convolution operation using Triton optimizations.

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
        
        # Initialize weights and biases
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
        # Use Triton implementation
        return triton_conv1d_transpose(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )

# Keep original class for comparison
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