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
    bias_ptr,
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
    stride_depth,
    stride_height,
    stride_width,
    padding_depth,
    padding_height,
    padding_width,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate group information
    group_idx = out_ch_idx // group_size
    group_offset = group_idx * group_size
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, (1, 1, input_depth, input_height, input_width), [0, 0, 0, 0, 0])
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates
                input_d = out_d_idx * stride_depth - padding_depth + kd
                input_h = out_h_idx * stride_height - padding_height + kh
                input_w = out_w_idx * stride_width - padding_width + kw
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and 
                    input_h >= 0 and input_h < input_height and 
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input index
                    input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) + \
                               (input_d * input_height * input_width + input_h * input_width + input_w) * in_channels + \
                               (out_ch_idx % group_size)
                    
                    # Calculate weight index
                    weight_idx = (out_ch_idx * group_size + (out_ch_idx % group_size)) * kernel_depth * kernel_height * kernel_width + \
                                kd * kernel_height * kernel_width + kh * kernel_width + kw
                    
                    # Load values and accumulate
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_ch_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Calculate output index
    output_idx = batch_idx * (out_channels * output_depth * output_height * output_width) + \
                 (out_d_idx * output_height * output_width + out_h_idx * output_width + out_w_idx) * out_channels + \
                 out_ch_idx
    
    # Store result
    tl.store(output_ptr + output_idx, acc, mask=True)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Set up kernel parameters
        self.kernel_depth, self.kernel_height, self.kernel_width = kernel_size
        self.stride_depth, self.stride_height, self.stride_width = stride
        self.padding_depth, self.padding_height, self.padding_width = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get dimensions
        batch_size, _, input_depth, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_depth = (input_depth - 1) * self.stride_depth - 2 * self.padding_depth + self.kernel_depth + self.output_padding[0]
        output_height = (input_height - 1) * self.stride_height - 2 * self.padding_height + self.kernel_height + self.output_padding[1]
        output_width = (input_width - 1) * self.stride_width - 2 * self.padding_width + self.kernel_width + self.output_padding[2]
        
        # Prepare output tensor
        output = torch.empty(batch_size, self.out_channels, output_depth, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        if self.bias is not None:
            bias = self.bias.contiguous()
        else:
            bias = None
        
        # Launch kernel
        grid = (
            batch_size,
            self.out_channels,
            output_depth,
            output_height,
            output_width
        )
        
        # Configure block size
        BLOCK_SIZE = 128
        
        # Launch the kernel
        conv_transpose3d_kernel[grid](
            x,
            weight,
            output,
            bias,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            self.kernel_depth,
            self.kernel_height,
            self.kernel_width,
            self.stride_depth,
            self.stride_height,
            self.stride_width,
            self.padding_depth,
            self.padding_height,
            self.padding_width,
            self.groups,
            self.out_channels // self.groups,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output