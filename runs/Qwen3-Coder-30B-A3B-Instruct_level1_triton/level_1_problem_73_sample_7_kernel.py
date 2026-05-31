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
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    groups,
    group_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    out_h_idx = tl.program_id(3)
    out_w_idx = tl.program_id(4)
    
    # Calculate group info
    group_idx = out_c_idx // group_size
    out_c_local = out_c_idx % group_size
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input coordinates
                input_d = out_d_idx * stride_d + kd - pad_d
                input_h = out_h_idx * stride_h + kh - pad_h
                input_w = out_w_idx * stride_w + kw - pad_w
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_idx * (in_channels * input_depth * input_height * input_width) +
                                       group_idx * (group_size * input_depth * input_height * input_width) +
                                       input_d * (input_height * input_width) +
                                       input_h * input_width +
                                       input_w, 
                                       mask=True)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        out_c_idx * (group_size * kernel_depth * kernel_height * kernel_width) +
                                        out_c_local * (kernel_depth * kernel_height * kernel_width) +
                                        kd * (kernel_height * kernel_width) +
                                        kh * kernel_width +
                                        kw, 
                                        mask=True)
                    
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c_idx, mask=True)
        acc += bias_val
    
    # Store result
    output_offset = batch_idx * (out_channels * output_depth * output_height * output_width) + \
                   out_c_idx * (output_depth * output_height * output_width) + \
                   out_d_idx * (output_height * output_width) + \
                   out_h_idx * output_width + \
                   out_w_idx
    
    tl.store(output_ptr + output_offset, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, groups):
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + kernel_depth
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + kernel_height
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + kernel_width
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size
    BLOCK_SIZE = 16
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width
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
        in_channels // groups,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.groups = groups
        self.bias = bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize with Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.weight)
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call our Triton-based implementation
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.groups
        )

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, groups={self.groups}, bias={self.bias is not None}'