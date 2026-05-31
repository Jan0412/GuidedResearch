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
    out_d = tl.program_id(2)
    out_h = tl.program_id(3)
    out_w = tl.program_id(4)
    
    # Calculate which group this output channel belongs to
    group_id = out_ch_idx // group_size
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
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
                    
                    # Calculate input index
                    input_idx = (batch_idx * (in_channels * input_depth * input_height * input_width) +
                                (group_id * group_size + (out_ch_idx % group_size)) * (input_depth * input_height * input_width) +
                                input_d * (input_height * input_width) +
                                input_h * input_width +
                                input_w)
                    
                    # Calculate weight index
                    weight_idx = (out_ch_idx * (groups * kernel_depth * kernel_height * kernel_width) +
                                 group_id * (kernel_depth * kernel_height * kernel_width) +
                                 kd * (kernel_height * kernel_width) +
                                 kh * kernel_width +
                                 kw)
                    
                    # Load input value
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_ch_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Calculate output index
    output_idx = (batch_idx * (out_channels * output_depth * output_height * output_width) +
                  out_ch_idx * (output_depth * output_height * output_width) +
                  out_d * (output_height * output_width) +
                  out_h * output_width +
                  out_w)
    
    # Store result
    tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of 3D transposed convolution
    """
    # Get input dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth + output_padding[0]
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height + output_padding[1]
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width + output_padding[2]
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Set up grid
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
    )
    
    # Group size
    group_size = out_channels // groups
    
    # Launch kernel
    if bias is not None:
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
            group_size,
            BLOCK_SIZE=128
        )
    else:
        conv_transpose3d_kernel[grid](
            input_tensor,
            weight,
            output,
            None,
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
            group_size,
            BLOCK_SIZE=128
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
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution using Triton kernel
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