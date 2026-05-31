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
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    group_idx = tl.program_id(2)
    
    # Calculate output dimensions
    out_h = height_out
    out_w = width_out
    
    # Shared memory for weight tiles
    tile_h = min(BLOCK_SIZE, out_h)
    tile_w = min(BLOCK_SIZE, out_w)
    
    # Initialize accumulator
    acc = tl.zeros((tile_h, tile_w), dtype=tl.float32)
    
    # Loop over input channels and groups
    for ch_idx in range(0, in_channels, GROUP_SIZE):
        # Load weight tile
        weight_tile = tl.load(weight_ptr + 
                             (out_ch_idx * in_channels + ch_idx) * kernel_h * kernel_w +
                             tl.arange(0, kernel_h)[:, None] * kernel_w + 
                             tl.arange(0, kernel_w)[None, :])
        
        # Load input tile
        input_tile = tl.load(input_ptr + 
                            (batch_idx * in_channels + ch_idx) * height_in * width_in +
                            tl.arange(0, height_in)[:, None] * width_in + 
                            tl.arange(0, width_in)[None, :])
        
        # Perform convolution
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate output position
                oh = kh * stride_h - padding_h
                ow = kw * stride_w - padding_w
                
                # Apply output padding
                oh += output_padding_h
                ow += output_padding_w
                
                # Check bounds
                if oh >= 0 and oh < out_h and ow >= 0 and ow < out_w:
                    # Update accumulator
                    acc[oh, ow] += input_tile[oh, ow] * weight_tile[kh, kw]
    
    # Store output
    output_offset = batch_idx * out_channels * out_h * out_w + out_ch_idx * out_h * out_w
    tl.store(output_ptr + output_offset + 
             tl.arange(0, tile_h)[:, None] * out_w + 
             tl.arange(0, tile_w)[None, :], acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    output_padding_h, output_padding_w = output_padding
    
    height_out = (height_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
    width_out = (width_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        groups
    )
    
    # Launch kernel
    BLOCK_SIZE = 16
    GROUP_SIZE = 16
    
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with asymmetric input and a square kernel.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        """
        # Convert to float32 if needed
        if x.dtype != torch.float32:
            x = x.float()
            
        # Use Triton kernel for computation
        output = triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )
        return output

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