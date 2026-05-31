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
    pad_d,
    pad_h,
    pad_w,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate group info
    group_idx = out_c_idx // group_size
    local_c_idx = out_c_idx % group_size
    
    # Calculate output position
    out_d = out_d_idx
    out_h = out_h_idx
    out_w = out_w_idx
    
    # Calculate corresponding input positions
    input_d_start = out_d * stride_d - pad_d
    input_h_start = out_h * stride_h - pad_h
    input_w_start = out_w * stride_w - pad_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates
                input_d = input_d_start + kd
                input_h = input_h_start + kh
                input_w = input_w_start + kw
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input index
                    input_idx = (batch_idx * (in_channels * input_depth * input_height * input_width) +
                                (group_idx * group_size + local_c_idx) * (input_depth * input_height * input_width) +
                                input_d * (input_height * input_width) +
                                input_h * input_width +
                                input_w)
                    
                    # Calculate weight index
                    weight_idx = (out_c_idx * (group_size * kernel_depth * kernel_height * kernel_width) +
                                 local_c_idx * (kernel_depth * kernel_height * kernel_width) +
                                 kd * (kernel_height * kernel_width) +
                                 kh * kernel_width +
                                 kw)
                    
                    # Load values and accumulate
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    acc += input_val * weight_val
    
    # Write output
    output_idx = (batch_idx * (out_channels * output_depth * output_height * output_width) +
                 out_c_idx * (output_depth * output_height * output_width) +
                 out_d * (output_height * output_width) +
                 out_h * output_width +
                 out_w)
    
    tl.store(output_ptr + output_idx, acc, mask=True)

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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride - 2 * self.padding + self.kernel_size
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        # Define kernel parameters
        kernel_depth = self.kernel_size
        kernel_height = self.kernel_size
        kernel_width = self.kernel_size
        
        # Grid configuration
        grid = (
            batch_size,
            self.out_channels,
            output_depth,
            output_height,
            output_width
        )
        
        # Launch kernel
        BLOCK_SIZE = 1024
        GROUP_SIZE = 32
        
        # Call the Triton kernel
        conv_transpose3d_kernel[grid](
            x,
            weight,
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
            kernel_depth,
            kernel_height,
            kernel_width,
            self.stride,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.padding,
            self.groups,
            in_channels // self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUP_SIZE=GROUP_SIZE
        )
        
        # Add bias if present
        if self.bias is not None:
            bias_expanded = self.bias.view(1, -1, 1, 1, 1)
            output = output + bias_expanded
            
        return output