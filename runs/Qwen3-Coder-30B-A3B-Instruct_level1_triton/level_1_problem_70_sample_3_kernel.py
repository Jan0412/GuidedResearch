import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate group information
    group_size = out_channels // groups
    group_idx = out_c_idx // group_size
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                input_d = out_d_idx * stride_d - padding_d + kd * dilation_d
                input_h = out_h_idx * stride_h - padding_h + kh * dilation_h
                input_w = out_w_idx * stride_w - padding_w + kw * dilation_w
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input index
                    input_idx = (batch_idx * (in_channels * input_depth * input_height * input_width) +
                                (group_idx * group_size + (out_c_idx % group_size)) * (input_depth * input_height * input_width) +
                                input_d * (input_height * input_width) +
                                input_h * input_width +
                                input_w)
                    
                    # Calculate weight index
                    weight_idx = (group_idx * group_size + (out_c_idx % group_size)) * (kernel_depth * kernel_height * kernel_width * in_channels) +
                                kd * (kernel_height * kernel_width * in_channels) +
                                kh * (kernel_width * in_channels) +
                                kw * in_channels +
                                (out_c_idx % group_size)
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    output_idx = (batch_idx * (out_channels * output_depth * output_height * output_width) +
                  out_c_idx * (output_depth * output_height * output_width) +
                  out_d_idx * (output_height * output_width) +
                  out_h_idx * output_width +
                  out_w_idx)
    
    tl.store(output_ptr + output_idx, acc, mask=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Launch kernel
        grid = (
            batch_size,
            self.out_channels,
            output_depth,
            output_height,
            output_width
        )
        
        # Define block sizes
        BLOCK_SIZE = 256
        GROUPS_PER_BLOCK = 1
        
        # Launch kernel
        conv_transpose3d_kernel[grid](
            x,
            self.weight,
            output,
            batch_size,
            in_channels,
            self.out_channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.dilation,
            self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS_PER_BLOCK=GROUPS_PER_BLOCK
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output