import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def conv3d_kernel(
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
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    dilation_d,
    dilation_h,
    dilation_w,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate global output index
    output_idx = (
        batch_idx * (out_channels * output_depth * output_height * output_width) +
        out_ch_idx * (output_depth * output_height * output_width) +
        out_d_idx * (output_height * output_width) +
        out_h_idx * output_width +
        out_w_idx
    )
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Group handling
    group_idx = out_ch_idx // (out_channels // groups)
    weight_offset = group_idx * (in_channels // groups) * kernel_depth * kernel_height * kernel_width
    
    # Loop over input channels and kernel dimensions
    for k_d in range(kernel_depth):
        for k_h in range(kernel_height):
            for k_w in range(kernel_width):
                # Calculate input position with stride and padding
                d = out_d_idx * stride_d - padding_d + k_d * dilation_d
                h = out_h_idx * stride_h - padding_h + k_h * dilation_h
                w = out_w_idx * stride_w - padding_w + k_w * dilation_w
                
                # Check bounds
                if d >= 0 and d < input_depth and h >= 0 and h < input_height and w >= 0 and w < input_width:
                    # Calculate input index
                    input_ch_start = group_idx * (in_channels // groups)
                    input_idx = (
                        batch_idx * (in_channels * input_depth * input_height * input_width) +
                        input_ch_start * (input_depth * input_height * input_width) +
                        d * (input_height * input_width) +
                        h * input_width +
                        w
                    )
                    
                    # Calculate weight index
                    weight_idx = (
                        out_ch_idx * (in_channels // groups * kernel_depth * kernel_height * kernel_width) +
                        (k_d * kernel_height * kernel_width + k_h * kernel_width + k_w)
                    )
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx + weight_offset, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx, mask=True)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + output_idx, acc, mask=True)


def triton_conv3d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    groups: int = 1
):
    """
    Triton implementation of 3D convolution
    """
    # Ensure inputs are on GPU
    input_tensor = input_tensor.cuda()
    weight = weight.cuda()
    if bias is not None:
        bias = bias.cuda()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_height = (input_height + 2 * padding[1] - (dilation[1] * (kernel_height - 1) + 1)) // stride[1] + 1
    output_width = (input_width + 2 * padding[2] - (dilation[2] * (kernel_width - 1) + 1)) // stride[2] + 1
    
    # Create output tensor
    output = torch.empty(
        batch_size, out_channels, output_depth, output_height, output_width,
        dtype=torch.float32, device='cuda'
    )
    
    # Define block sizes
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 16
    
    # Grid configuration
    grid = (
        batch_size,           # batch dimension
        out_channels,         # output channels
        output_depth,         # output depth
        output_height,        # output height
        output_width          # output width
    )
    
    # Launch kernel
    conv3d_kernel[grid](
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
        dilation[0],
        dilation[1],
        dilation[2],
        groups,
        in_channels // groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, stride={self.stride}, '
            f'padding={self.padding}, dilation={self.dilation}, '
            f'groups={self.groups}, bias={self.bias is not None}'
        )