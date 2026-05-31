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
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_d_id = tl.program_id(2)
    
    # Calculate group info
    ch_per_group = out_channels // groups
    group_id = out_ch_id // ch_per_group
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                d_in = out_d_id * stride_d + kd - padding_d
                h_in = out_d_id * stride_h + kh - padding_h
                w_in = out_d_id * stride_w + kw - padding_w
                
                # Check bounds
                if (d_in >= 0 and d_in < input_depth and 
                    h_in >= 0 and h_in < input_height and 
                    w_in >= 0 and w_in < input_width):
                    
                    # Calculate input index
                    input_idx = (
                        batch_id * (in_channels * input_depth * input_height * input_width) +
                        group_id * (input_depth * input_height * input_width) +
                        d_in * (input_height * input_width) +
                        h_in * input_width +
                        w_in
                    )
                    
                    # Calculate weight index
                    weight_idx = (
                        out_ch_id * (in_channels // groups * kernel_depth * kernel_height * kernel_width) +
                        (kd * kernel_height * kernel_width + kh * kernel_width + kw)
                    )
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    if out_d_id < output_depth:
        output_idx = (
            batch_id * (out_channels * output_depth * output_height * output_width) +
            out_ch_id * (output_depth * output_height * output_width) +
            out_d_id * (output_height * output_width)
        )
        tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), groups=1):
    """
    Triton implementation of 3D transposed convolution
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    
    output_depth = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth
    output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height
    output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    grid = (
        batch_size,
        out_channels,
        output_depth
    )
    
    # Launch kernel
    conv_transpose3d_kernel[grid](
        input_tensor,
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
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        groups,
        BLOCK_SIZE=32,
        GROUP_SIZE=8
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
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
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            groups=self.groups
        )

# For compatibility with original interface
def get_inputs():
    batch_size = 4
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    depth = 32
    height = 64
    width = 128
    stride = 2
    padding = 1
    groups = 4
    
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [32, 32, 3, 2, 1, 0, 4, False]