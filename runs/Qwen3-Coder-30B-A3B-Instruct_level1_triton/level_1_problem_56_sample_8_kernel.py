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
    input_shape,
    weight_shape,
    output_shape,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    batch_size,
    in_channels,
    out_channels,
    input_h,
    input_w,
    output_h,
    output_w,
    weight_h,
    weight_w,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_c_idx = tl.program_id(2)
    
    # Calculate which output channel this thread handles
    channels_per_group = out_channels // groups
    group_offset = group_idx * channels_per_group
    channel_offset = group_offset + (out_c_idx % channels_per_group)
    
    # Shared memory for input tile
    tile_size = BLOCK_SIZE
    input_tile = tl.shared_tile(input_ptr, (tile_size, tile_size))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for c in range(in_channels // groups):
        for kh in range(weight_h):
            for kw in range(weight_w):
                # Calculate input positions
                ih = tl.arange(0, tile_size) * stride_h - padding_h + kh * dilation_h
                iw = tl.arange(0, tile_size) * stride_w - padding_w + kw * dilation_w
                
                # Bounds checking
                ih_valid = (ih >= 0) & (ih < input_h)
                iw_valid = (iw >= 0) & (iw < input_w)
                valid_mask = ih_valid & iw_valid
                
                # Load input data
                input_data = tl.load(input_ptr + batch_idx * (in_channels * input_h * input_w) +
                                   (c + group_idx * (in_channels // groups)) * (input_h * input_w) +
                                   ih[:, None] * input_w + iw[None, :], mask=valid_mask[:, None])
                
                # Load weight data
                weight_data = tl.load(weight_ptr + channel_offset * (in_channels // groups * weight_h * weight_w) +
                                    c * (weight_h * weight_w) + kh * weight_w + kw)
                
                # Accumulate
                acc += tl.sum(input_data * weight_data)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel_offset)
        acc += bias_val
    
    # Store output
    if batch_idx < batch_size and out_c_idx < out_channels:
        tl.store(output_ptr + batch_idx * (out_channels * output_h * output_w) +
                out_c_idx * (output_h * output_w) +
                tl.arange(0, output_h)[:, None] * output_w + tl.arange(0, output_w)[None, :],
                acc, mask=(tl.arange(0, output_h)[:, None] < output_h) &
                         (tl.arange(0, output_w)[None, :] < output_w))

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution
    """
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_h, input_w = input_tensor.shape
    out_channels, _, weight_h, weight_w = weight.shape
    
    # Calculate output dimensions
    output_h = (input_h + 2 * padding[0] - (dilation[0] * (weight_h - 1) + 1)) // stride[0] + 1
    output_w = (input_w + 2 * padding[1] - (dilation[1] * (weight_w - 1) + 1)) // stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_h, output_w, device=input_tensor.device, dtype=torch.float32)
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 16
    GROUP_SIZE = 8
    
    # Grid configuration
    grid = (
        batch_size,
        groups,
        out_channels
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.shape,
        weight.shape,
        output.shape,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        output_h,
        output_w,
        weight_h,
        weight_w,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.
    Optimized using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(x, self.weight, self.bias, 
                           stride=self.stride, 
                           padding=self.padding, 
                           dilation=self.dilation, 
                           groups=self.groups)