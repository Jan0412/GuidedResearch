import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_row = tl.program_id(2)
    out_col = tl.program_id(3)
    
    # Calculate group info
    ch_per_group = in_channels // groups
    out_ch_per_group = out_channels // groups
    group_idx = out_ch_idx // out_ch_per_group
    
    # Shared memory for input tile
    tile_size = BLOCK_SIZE + 2 * padding_h + 2 * padding_w
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(tile_size, tile_size))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate output position
    out_h = out_row
    out_w = out_col
    
    # Calculate input start positions
    in_start_h = out_h * stride_h - padding_h
    in_start_w = out_w * stride_w - padding_w
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input coordinates
            in_h = in_start_h + kh * dilation_h
            in_w = in_start_w + kw * dilation_w
            
            # Load weight
            weight_offset = group_idx * out_ch_per_group * ch_per_group * kernel_height * kernel_width
            weight_idx = out_ch_idx * ch_per_group * kernel_height * kernel_width + \
                        (kh * kernel_width + kw) * ch_per_group
            
            # Load input if within bounds
            if in_h >= 0 and in_h < input_height and in_w >= 0 and in_w < input_width:
                # Calculate input index for this channel group
                input_offset = batch_idx * in_channels * input_height * input_width
                input_idx = input_offset + group_idx * ch_per_group * input_height * input_width
                
                # Load input value
                input_val = tl.load(input_ptr + input_idx + 
                                  (in_h * input_width + in_w) * ch_per_group + 
                                  (kh * kernel_width + kw) % ch_per_group, 
                                  mask=True, other=0.0)
                
                # Load weight
                weight_val = tl.load(weight_ptr + weight_idx + 
                                   (kh * kernel_width + kw) % ch_per_group, 
                                   mask=True, other=0.0)
                
                acc += input_val * weight_val
            else:
                # Out of bounds - skip
                pass
    
    # Add bias if exists
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx, mask=True, other=0.0)
        acc += bias_val
    
    # Write output
    if batch_idx < batch_size and out_row < output_height and out_col < output_width:
        output_idx = batch_idx * out_channels * output_height * output_width + \
                    out_ch_idx * output_height * output_width + \
                    out_row * output_width + out_col
        tl.store(output_ptr + output_idx, acc[0], mask=True)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Custom Triton implementation of 2D convolution
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 16
    GROUP_SIZE = 4
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_height,
        output_width
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Extract parameters from the original conv layer
        weight = self.conv2d.weight.data
        bias = self.conv2d.bias.data if self.conv2d.bias is not None else None
        stride = self.conv2d.stride
        padding = self.conv2d.padding
        dilation = self.conv2d.dilation
        groups = self.conv2d.groups
        
        # Use Triton kernel for convolution
        return triton_conv2d(x, weight, bias, stride, padding, dilation, groups)