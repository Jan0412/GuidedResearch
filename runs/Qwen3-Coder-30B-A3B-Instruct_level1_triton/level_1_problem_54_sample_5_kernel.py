import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
    dilation_d,
    dilation_w,
    dilation_h,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    TILE_SIZE_D: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
    TILE_SIZE_H: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    out_h_idx = tl.program_id(4)
    
    # Calculate global output indices
    out_d = out_d_idx * TILE_SIZE_D
    out_w = out_w_idx * TILE_SIZE_W
    out_h = out_h_idx * TILE_SIZE_H
    
    # Shared memory for input tile and weight tile
    input_tile = tl.shared_tensor(tl.float32, (TILE_SIZE_D, TILE_SIZE_W, TILE_SIZE_H, in_channels))
    weight_tile = tl.shared_tensor(tl.float32, (kernel_depth, kernel_width, kernel_height, in_channels, out_channels))
    
    # Initialize accumulator
    acc = tl.zeros((TILE_SIZE_D, TILE_SIZE_W, TILE_SIZE_H), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for k_d in range(0, kernel_depth):
        for k_w in range(0, kernel_width):
            for k_h in range(0, kernel_height):
                # Calculate input positions
                in_d = out_d * stride_d + k_d * dilation_d - padding_d
                in_w = out_w * stride_w + k_w * dilation_w - padding_w
                in_h = out_h * stride_h + k_h * dilation_h - padding_h
                
                # Check bounds for input
                if in_d >= 0 and in_d < input_depth and \
                   in_w >= 0 and in_w < input_width and \
                   in_h >= 0 and in_h < input_height:
                    
                    # Load input tile
                    input_data = tl.load(input_ptr + 
                        batch_idx * (in_channels * input_depth * input_width * input_height) +
                        in_d * (in_channels * input_width * input_height) +
                        in_w * (in_channels * input_height) +
                        in_h * in_channels +
                        tl.arange(0, in_channels), 
                        mask=(tl.arange(0, in_channels) < in_channels), 
                        other=0.0)
                    
                    # Load weight tile
                    weight_data = tl.load(weight_ptr + 
                        k_d * (in_channels * out_channels * kernel_width * kernel_height) +
                        k_w * (in_channels * out_channels * kernel_height) +
                        k_h * (in_channels * out_channels) +
                        tl.arange(0, in_channels) * out_channels +
                        out_ch_idx,
                        mask=(tl.arange(0, in_channels) < in_channels),
                        other=0.0)
                    
                    # Accumulate
                    acc += tl.expand_dims(input_data, axis=0) * weight_data
    
    # Write output
    output_offset = batch_idx * (out_channels * output_depth * output_width * output_height) + \
                   out_ch_idx * (output_depth * output_width * output_height) + \
                   out_d_idx * (output_width * output_height) + \
                   out_w_idx * output_height + \
                   out_h_idx
                   
    output_mask = (out_d + tl.arange(0, TILE_SIZE_D) < output_depth) & \
                  (out_w + tl.arange(0, TILE_SIZE_W) < output_width) & \
                  (out_h + tl.arange(0, TILE_SIZE_H) < output_height)
    
    tl.store(output_ptr + output_offset, acc, mask=output_mask)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth + 2 * padding[0] - (dilation[0] * (kernel_depth - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    output_height = (input_height + 2 * padding[2] - (dilation[2] * (kernel_height - 1) + 1)) // stride[2] + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    TILE_SIZE_D = 8
    TILE_SIZE_W = 8
    TILE_SIZE_H = 8
    BLOCK_SIZE = 256
    
    # Grid dimensions
    grid = (
        batch_size,
        out_channels,
        (output_depth + TILE_SIZE_D - 1) // TILE_SIZE_D,
        (output_width + TILE_SIZE_W - 1) // TILE_SIZE_W,
        (output_height + TILE_SIZE_H - 1) // TILE_SIZE_H
    )
    
    # Launch kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
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
        dilation[0],
        dilation[1],
        dilation[2],
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=8,
        TILE_SIZE_D=TILE_SIZE_D,
        TILE_SIZE_W=TILE_SIZE_W,
        TILE_SIZE_H=TILE_SIZE_H
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            dilation=(self.dilation, self.dilation, self.dilation)
        )