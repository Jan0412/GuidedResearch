import torch
import torch.nn as nn
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
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Calculate output dimensions
    kernel_size = kernel_depth * kernel_height * kernel_width
    
    # Shared memory for weight cache
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_PER_BLOCK, CHANNELS_PER_BLOCK, kernel_depth, kernel_height, kernel_width))
    
    # Calculate input indices
    input_d_start = out_d * stride_d - padding_d
    input_h_start = out_h * stride_h - padding_h
    input_w_start = out_w * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((CHANNELS_PER_BLOCK,), dtype=tl.float32)
    
    # Process over groups
    for g in range(0, groups, GROUPS_PER_BLOCK):
        # Load weights for this group
        if g + group_idx < groups:
            for c in range(0, CHANNELS_PER_BLOCK):
                if c < in_channels // groups:
                    for kd in range(kernel_depth):
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):
                                weight_val = tl.load(weight_ptr + 
                                                    (g + group_idx) * (in_channels // groups) * kernel_size +
                                                    c * kernel_size +
                                                    kd * kernel_height * kernel_width +
                                                    kh * kernel_width +
                                                    kw)
                                tl.store(shared_weight + 
                                        (group_idx % GROUPS_PER_BLOCK) * CHANNELS_PER_BLOCK * kernel_depth * kernel_height * kernel_width +
                                        (c % CHANNELS_PER_BLOCK) * kernel_depth * kernel_height * kernel_width +
                                        kd * kernel_height * kernel_width +
                                        kh * kernel_width +
                                        kw, weight_val)
        
        # Compute convolution for current group
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    input_d = input_d_start + kd
                    input_h = input_h_start + kh
                    input_w = input_w_start + kw
                    
                    # Check bounds
                    if (input_d >= 0 and input_d < input_depth and
                        input_h >= 0 and input_h < input_height and
                        input_w >= 0 and input_w < input_width):
                        
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_idx * (in_channels // groups) * input_depth * input_height * input_width +
                                          (g + group_idx) * (in_channels // groups) * input_depth * input_height * input_width +
                                          (input_d * input_height * input_width +
                                           input_h * input_width +
                                           input_w))
                        
                        # Load weight
                        weight_val = tl.load(shared_weight + 
                                           (group_idx % GROUPS_PER_BLOCK) * CHANNELS_PER_BLOCK * kernel_depth * kernel_height * kernel_width +
                                           (0 % CHANNELS_PER_BLOCK) * kernel_depth * kernel_height * kernel_width +
                                           kd * kernel_height * kernel_width +
                                           kh * kernel_width +
                                           kw)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Store output
    for c in range(0, CHANNELS_PER_BLOCK):
        if c < out_channels // groups:
            output_offset = (batch_idx * out_channels * output_depth * output_height * output_width +
                           (g + group_idx) * (out_channels // groups) * output_depth * output_height * output_width +
                           (out_d * output_height * output_width +
                            out_h * output_width +
                            out_w))
            tl.store(output_ptr + output_offset, acc[c])

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
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dimensions
        batch_size, in_channels, input_depth, input_height, input_width = x.shape
        out_channels, _, kernel_depth, kernel_height, kernel_width = self.weight.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        output_height = (input_height - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        output_width = (input_width - 1) * self.stride + self.kernel_size - 2 * self.padding + self.output_padding
        
        # Create output tensor
        output = torch.zeros(batch_size, out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Convert to appropriate data types for kernel
        x = x.contiguous().to(torch.float32)
        weight = self.weight.contiguous().to(torch.float32)
        
        # Launch kernel
        grid = (
            batch_size,
            self.groups,
            output_depth,
            output_height,
            output_width
        )
        
        # Define block sizes
        BLOCK_SIZE = 128
        GROUPS_PER_BLOCK = 1
        CHANNELS_PER_BLOCK = 1
        
        # Launch kernel with appropriate parameters
        conv_transpose3d_kernel[grid](
            x,
            weight,
            output,
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
            self.stride,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.padding,
            self.groups,
            BLOCK_SIZE=BLOCK_SIZE,
            GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
            CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)
            
        return output