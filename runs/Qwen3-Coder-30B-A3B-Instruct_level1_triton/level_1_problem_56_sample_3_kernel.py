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
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_height_stride,
    weight_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
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
    group_size,
    has_bias,
    BLOCK_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_C: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    
    # Calculate output width tile
    out_x = tl.program_id(3)
    
    # Shared memory for input tile
    input_tile = tl.shared_pointer(
        tl.zeros((TILE_H + 2 * padding_h, TILE_W + 2 * padding_w), dtype=tl.float32),
        TILE_H + 2 * padding_h,
        TILE_W + 2 * padding_w
    )
    
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific indices
        group_start_c = g * group_size
        group_end_c = (g + 1) * group_size
        
        # Loop over input channels in tiles
        for c in range(0, group_size, TILE_C):
            # Load weight tile
            weight_tile = tl.zeros((TILE_C, kernel_height, kernel_width), dtype=tl.float32)
            
            # Load input tile (with padding)
            input_tile = tl.zeros((TILE_H + 2 * padding_h, TILE_W + 2 * padding_w), dtype=tl.float32)
            
            # Load input data into shared memory
            for i in range(TILE_H + 2 * padding_h):
                for j in range(TILE_W + 2 * padding_w):
                    if i < padding_h or i >= TILE_H + padding_h or j < padding_w or j >= TILE_W + padding_w:
                        input_tile[i, j] = 0.0
                    else:
                        h = out_y * stride_h + i - padding_h
                        w = out_x * stride_w + j - padding_w
                        
                        if h >= 0 and h < input_height and w >= 0 and w < input_width:
                            input_val = tl.load(input_ptr + 
                                              batch_idx * input_batch_stride +
                                              (group_start_c + c) * input_channel_stride +
                                              h * input_height_stride +
                                              w * input_width_stride)
                            input_tile[i, j] = input_val
                        else:
                            input_tile[i, j] = 0.0
            
            # Compute convolution for this tile
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Load weight
                    weight_val = tl.load(weight_ptr + 
                                       out_channel_idx * weight_out_channel_stride +
                                       (group_start_c + c) * weight_in_channel_stride +
                                       kh * weight_height_stride +
                                       kw * weight_width_stride)
                    
                    # Apply dilation
                    for i in range(TILE_H):
                        for j in range(TILE_W):
                            acc[i, j] += input_tile[i + kh * dilation_h, j + kw * dilation_w] * weight_val
    
    # Write output
    for i in range(TILE_H):
        for j in range(TILE_W):
            if out_y * TILE_H + i < output_height and out_x * TILE_W + j < output_width:
                out_idx = batch_idx * output_batch_stride + \
                         out_channel_idx * output_channel_stride + \
                         (out_y * TILE_H + i) * output_height_stride + \
                         (out_x * TILE_W + j) * output_width_stride
                
                val = acc[i, j]
                
                if has_bias:
                    bias_val = tl.load(bias_ptr + out_channel_idx)
                    val += bias_val
                    
                tl.store(output_ptr + out_idx, val)

def triton_conv2d(input_tensor, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Custom Triton implementation of 2D convolution.
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Set up strides
    input_batch_stride = in_channels * input_height * input_width
    input_channel_stride = input_height * input_width
    input_height_stride = input_width
    input_width_stride = 1
    
    weight_out_channel_stride = in_channels * kernel_height * kernel_width
    weight_in_channel_stride = kernel_height * kernel_width
    weight_height_stride = kernel_width
    weight_width_stride = 1
    
    output_batch_stride = out_channels * output_height * output_width
    output_channel_stride = output_height * output_width
    output_height_stride = output_width
    output_width_stride = 1
    
    # Launch configuration
    TILE_H = 16
    TILE_W = 16
    TILE_C = 8
    
    grid = (
        batch_size,  # batch dimension
        out_channels,  # output channel dimension
        (output_height + TILE_H - 1) // TILE_H,  # output height tiles
        (output_width + TILE_W - 1) // TILE_W   # output width tiles
    )
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_batch_stride,
        input_channel_stride,
        input_height_stride,
        input_width_stride,
        weight_out_channel_stride,
        weight_in_channel_stride,
        weight_height_stride,
        weight_width_stride,
        output_batch_stride,
        output_channel_stride,
        output_height_stride,
        output_width_stride,
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
        in_channels // groups,
        bias is not None,
        BLOCK_SIZE=1024,
        TILE_H=TILE_H,
        TILE_W=TILE_W,
        TILE_C=TILE_C
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)