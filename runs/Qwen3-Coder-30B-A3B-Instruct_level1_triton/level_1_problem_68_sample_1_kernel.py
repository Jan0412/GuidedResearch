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
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    output_padding_d,
    output_padding_w,
    output_padding_h,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_d_id = tl.program_id(2)
    
    # Calculate group information
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Calculate which group this thread block is responsible for
    group_id = out_ch_id // out_channels_per_group
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, shape=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for k_d in range(kernel_depth):
        for k_w in range(kernel_width):
            for k_h in range(kernel_height):
                # Calculate output coordinates
                out_d = out_d_id * stride_d - padding_d + k_d
                out_w = tl.program_id(3) * stride_w - padding_w + k_w
                out_h = tl.program_id(4) * stride_h - padding_h + k_h
                
                # Check bounds
                if (out_d >= 0 and out_d < input_depth and
                    out_w >= 0 and out_w < input_width and
                    out_h >= 0 and out_h < input_height):
                    
                    # Calculate input indices
                    input_idx = (
                        batch_id * (in_channels * input_depth * input_width * input_height) +
                        group_id * (channels_per_group * input_depth * input_width * input_height) +
                        (out_d * input_width * input_height + out_w * input_height + out_h)
                    )
                    
                    # Calculate weight indices
                    weight_idx = (
                        out_ch_id * (channels_per_group * kernel_depth * kernel_width * kernel_height) +
                        (k_d * kernel_width * kernel_height + k_w * kernel_height + k_h)
                    )
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_ch_id
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Store result
    if acc != 0.0:
        output_idx = (
            batch_id * (out_channels * output_depth * output_width * output_height) +
            out_ch_id * (output_depth * output_width * output_height) +
            (out_d_id * output_width * output_height + tl.program_id(3) * output_height + tl.program_id(4))
        )
        tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of ConvTranspose3d
    """
    # Extract dimensions
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1]
    output_height = (input_height - 1) * stride[2] - 2 * padding[2] + kernel_height + output_padding[2]
    
    # Create output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Handle bias
    if bias is not None:
        bias = bias.contiguous()
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_width,
        output_height
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_width,
        input_height,
        output_depth,
        output_width,
        output_height,
        kernel_depth,
        kernel_width,
        kernel_height,
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        output_padding[0],
        output_padding[1],
        output_padding[2],
        groups,
        BLOCK_SIZE=128,
        GROUP_SIZE=8
    )
    
    return output

class ModelNew(nn.Module):
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, stride={self.stride}, '
            f'padding={self.padding}, output_padding={self.output_padding}, '
            f'groups={self.groups}, bias={self.bias is not None}'
        )