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
    bias_enabled,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    group_idx = tl.program_id(3)
    
    # Calculate output dimensions
    tile_h_start = out_h_idx * BLOCK_SIZE_H
    tile_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_tensor(tl.make_block_ptr(input_ptr, 
                                                       shape=(batch_size, in_channels, input_height, input_width),
                                                       strides=(in_channels * input_height * input_width, input_height * input_width, input_width, 1),
                                                       offsets=(batch_idx, 0, 0, 0),
                                                       block_shape=(1, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
                                                       order=(3, 2, 1, 0)), 
                                    (1, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W))
    
    # Process output tile
    for c in range(0, out_channels, GROUP_SIZE_M):
        if c + group_idx >= out_channels:
            break
            
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        
        # Loop over kernel
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                ih = tile_h_start + kh * stride_h - padding_h
                iw = tile_w_start + kw * stride_w - padding_w
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Load input tile
                    input_tile = tl.load(shared_input + (0, 0, ih, iw))
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + 
                                       (c + group_idx, 0, kh, kw) +
                                       (group_idx * (out_channels // groups) * in_channels // groups * kernel_height * kernel_width, 0, 0, 0))
                    
                    # Multiply and accumulate
                    acc += input_tile * weight_val
                    
        # Store result
        if bias_enabled:
            bias_val = tl.load(bias_ptr + (c + group_idx,))
            acc += bias_val
        
        # Store output
        tl.store(output_ptr + 
                (batch_idx, c + group_idx, tile_h_start, tile_w_start),
                acc,
                mask=((tile_h_start + tl.arange(0, BLOCK_SIZE_H)) < output_height) &
                      ((tile_w_start + tl.arange(0, BLOCK_SIZE_W)) < output_width))

def triton_conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    # Input shapes
    batch_size, in_channels, input_height, input_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride - 2 * padding + kernel_height + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + kernel_width + output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input.device, dtype=torch.float32)
    
    # Define parameters
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_C = 16
    GROUP_SIZE_M = 16
    
    # Grid dimensions
    grid = (
        batch_size,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
        (out_channels + GROUP_SIZE_M - 1) // GROUP_SIZE_M
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input,
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
        stride,
        stride,
        padding,
        padding,
        groups,
        bias is not None,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output

class ModelNew(nn.Module):
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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

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