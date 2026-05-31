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
    bias_ptr,
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
    kernel_height,
    kernel_width,
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    groups,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_depth_idx = tl.program_id(2)
    
    # Calculate group info
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    group_idx = out_channel_idx // out_channels_per_group
    
    # Shared memory for input tile
    shared_input = tl.shared_tensor(tl.float32, (BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for k_d in range(kernel_depth):
        for k_h in range(kernel_height):
            for k_w in range(kernel_width):
                # Calculate output position
                out_d = out_depth_idx * stride_depth + k_d - padding_depth
                out_h = out_depth_idx * stride_height + k_h - padding_height
                out_w = out_depth_idx * stride_width + k_w - padding_width
                
                # Check bounds
                if (out_d >= 0 and out_d < input_depth and 
                    out_h >= 0 and out_h < input_height and 
                    out_w >= 0 and out_w < input_width):
                    
                    # Calculate input indices
                    input_idx = (
                        batch_idx * (in_channels * input_depth * input_height * input_width) +
                        group_idx * (channels_per_group * input_depth * input_height * input_width) +
                        (out_channel_idx % out_channels_per_group) * (input_depth * input_height * input_width) +
                        out_d * (input_height * input_width) +
                        out_h * input_width +
                        out_w
                    )
                    
                    # Calculate weight index
                    weight_idx = (
                        out_channel_idx * (channels_per_group * kernel_depth * kernel_height * kernel_width) +
                        (out_channel_idx % out_channels_per_group) * (kernel_depth * kernel_height * kernel_width) +
                        k_d * (kernel_height * kernel_width) +
                        k_h * kernel_width +
                        k_w
                    )
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store output
    if bias_enabled:
        bias_val = tl.load(bias_ptr + out_channel_idx, mask=True)
        acc += bias_val
    
    # Calculate output index
    output_idx = (
        batch_idx * (out_channels * output_depth * output_height * output_width) +
        out_channel_idx * (output_depth * output_height * output_width) +
        out_depth_idx * (output_height * output_width)
    )
    
    tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, groups, output_padding):
    """
    Triton implementation of 3D transposed convolution
    """
    # Extract dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Define grid
    grid = (
        batch_size,
        out_channels,
        output_depth
    )
    
    # Launch kernel
    BLOCK_SIZE = 128
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        groups,
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of 3 integers representing the kernel size in the form (depth, height, width).
        stride (tuple, optional): Tuple of 3 integers representing the stride in the form (depth, height, width). Defaults to (1, 1, 1).
        padding (tuple, optional): Tuple of 3 integers representing the padding in the form (depth, height, width). Defaults to (0, 0, 0).
        output_padding (tuple, optional): Tuple of 3 integers representing the output padding in the form (depth, height, width). Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth_in, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.groups, 
            self.output_padding
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_channels={self.in_channels}',
            f'out_channels={self.out_channels}',
            f'kernel_size={self.kernel_size}',
            f'stride={self.stride}',
            f'padding={self.padding}',
            f'output_padding={self.output_padding}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])