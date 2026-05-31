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
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_d = tl.program_id(2)
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Calculate output position
    out_pos = batch_idx * out_channels * output_depth * output_height * output_width + \
              group_idx * (out_channels // groups) * output_depth * output_height * output_width + \
              out_d * output_height * output_width + \
              out_h * output_width + \
              out_w
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(GROUP_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates
                input_d = out_d * stride_depth - padding_depth + kd
                input_h = out_h * stride_height - padding_height + kh
                input_w = out_w * stride_width - padding_width + kw
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input position
                    input_pos = batch_idx * in_channels * input_depth * input_height * input_width + \
                               group_idx * (in_channels // groups) * input_depth * input_height * input_width + \
                               input_d * input_height * input_width + \
                               input_h * input_width + \
                               input_w
                    
                    # Calculate weight position
                    weight_pos = group_idx * (out_channels // groups) * in_channels * kernel_depth * kernel_height * kernel_width + \
                                (out_d * output_height * output_width + out_h * output_width + out_w) * kernel_depth * kernel_height * kernel_width + \
                                kd * kernel_height * kernel_width + \
                                kh * kernel_width + \
                                kw
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_pos, mask=True)
                    weight_val = tl.load(weight_ptr + weight_pos, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_pos = group_idx * (out_channels // groups) + (out_d * output_height * output_width + out_h * output_width + out_w)
        acc += tl.load(bias_ptr + bias_pos, mask=True)
    
    # Store result
    tl.store(output_ptr + out_pos, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Triton implementation of 3D transposed convolution
    """
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 128
    GROUP_SIZE = 8
    
    # Create grid
    grid = (
        batch_size,           # batch dimension
        groups,               # group dimension
        output_depth,         # depth dimension
        output_height,        # height dimension
        output_width          # width dimension
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

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
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights with Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )