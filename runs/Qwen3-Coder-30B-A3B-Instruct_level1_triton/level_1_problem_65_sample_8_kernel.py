import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose2d_kernel(
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
    groups,
    output_padding_h,
    output_padding_w,
    BLOCK_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    CHANNELS_PER_TILE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate global output position
    out_h = out_h_idx * TILE_H
    out_w = out_w_idx * TILE_W
    
    # Shared memory for tile
    tile_input = tl.shared_memory(dtype=tl.float32, shape=(TILE_H, TILE_W))
    
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input coordinates
            ih = out_h * stride_h + kh - padding_h
            iw = out_w * stride_w + kw - padding_w
            
            # Check bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                # Load input tile
                input_tile = tl.load(input_ptr + 
                                   batch_idx * in_channels * input_height * input_width +
                                   (out_c_idx % (in_channels // groups)) * input_height * input_width +
                                   ih * input_width + iw)
                
                # Load weight
                weight_val = tl.load(weight_ptr + 
                                   out_c_idx * (in_channels // groups) * kernel_height * kernel_width +
                                   (out_c_idx % (in_channels // groups)) * kernel_height * kernel_width +
                                   kh * kernel_width + kw)
                
                # Accumulate
                acc += input_tile * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c_idx)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + 
             batch_idx * out_channels * output_height * output_width +
             out_c_idx * output_height * output_width +
             out_h * output_width + out_w,
             acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    output_padding_h, output_padding_w = output_padding
    
    output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h
    output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define grid
    grid = (
        batch_size,
        out_channels,
        (output_height + 15) // 16,
        (output_width + 15) // 16
    )
    
    # Launch kernel
    TILE_H = 16
    TILE_W = 16
    CHANNELS_PER_TILE = 1
    
    conv_transpose2d_kernel[grid](
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
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        groups,
        output_padding_h,
        output_padding_w,
        BLOCK_SIZE=1024,
        TILE_H=TILE_H,
        TILE_W=TILE_W,
        CHANNELS_PER_TILE=CHANNELS_PER_TILE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        
        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )

# Keep the original test functions for compatibility
def get_inputs():
    batch_size = 8
    in_channels = 64
    out_channels = 64
    kernel_size = (3, 7)
    width = 512
    height = 512
    
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    batch_size = 8
    in_channels = 64
    out_channels = 64
    kernel_size = (3, 7)
    width = 512
    height = 512
    
    return [in_channels, out_channels, kernel_size]