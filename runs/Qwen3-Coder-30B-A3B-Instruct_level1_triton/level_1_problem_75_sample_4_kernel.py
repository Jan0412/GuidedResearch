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
    dilation_h,
    dilation_w,
    groups,
    channels_per_group,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE_H: tl.constexpr,
    OUTPUT_BLOCK_SIZE_W: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_h_start = tl.program_id(2) * OUTPUT_BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * OUTPUT_BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(OUTPUT_BLOCK_SIZE_H + 2 * padding_h, OUTPUT_BLOCK_SIZE_W + 2 * padding_w))
    
    # Load bias if available
    bias_val = tl.load(bias_ptr + group_idx * channels_per_group + tl.arange(0, channels_per_group)) if bias_ptr is not None else tl.zeros((channels_per_group,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input positions
            for c in range(channels_per_group):
                # For each output position in the block
                for oh in range(OUTPUT_BLOCK_SIZE_H):
                    for ow in range(OUTPUT_BLOCK_SIZE_W):
                        out_h = out_h_start + oh
                        out_w = out_w_start + ow
                        
                        # Check bounds
                        if out_h >= output_height or out_w >= output_width:
                            continue
                            
                        # Calculate corresponding input position
                        input_h = out_h * stride_h - padding_h + kh * dilation_h
                        input_w = out_w * stride_w - padding_w + kw * dilation_w
                        
                        # Check if input position is valid
                        if input_h >= 0 and input_h < input_height and input_w >= 0 and input_w < input_width:
                            # Load input value
                            input_val = tl.load(input_ptr + 
                                              batch_idx * (in_channels * input_height * input_width) +
                                              group_idx * channels_per_group * input_height * input_width +
                                              c * input_height * input_width +
                                              input_h * input_width +
                                              input_w)
                            
                            # Load weight
                            weight_val = tl.load(weight_ptr + 
                                               group_idx * (channels_per_group * kernel_height * kernel_width * out_channels) +
                                               c * (kernel_height * kernel_width * out_channels) +
                                               kh * (kernel_width * out_channels) +
                                               kw * out_channels +
                                               (out_h * output_width + out_w) % out_channels)
                            
                            # Accumulate
                            if out_h < output_height and out_w < output_width:
                                output_offset = batch_idx * (out_channels * output_height * output_width) + \
                                              group_idx * channels_per_group * output_height * output_width + \
                                              c * output_height * output_width + \
                                              out_h * output_width + out_w
                                
                                tl.atomic_add(output_ptr + output_offset, input_val * weight_val)

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_width - 1) + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Calculate parameters
    channels_per_group = in_channels // groups
    channels_per_output_group = out_channels // groups
    
    # Configure grid
    grid = (
        batch_size,
        groups,
        triton.cdiv(output_height, 16),
        triton.cdiv(output_width, 16)
    )
    
    # Launch kernel
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
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        channels_per_group,
        BLOCK_SIZE=1024,
        OUTPUT_BLOCK_SIZE_H=16,
        OUTPUT_BLOCK_SIZE_W=16
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

# For backward compatibility, keep the original class name
Model = ModelNew