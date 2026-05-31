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
    input_width,
    input_height,
    output_depth,
    output_width,
    output_height,
    kernel_depth,
    kernel_width,
    kernel_height,
    stride_d,
    stride_w,
    stride_h,
    padding_d,
    padding_w,
    padding_h,
    output_padding_d,
    output_padding_w,
    output_padding_h,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_d_idx = tl.program_id(2)
    
    # Calculate group information
    ch_per_group = out_channels // groups
    group_idx = out_ch_idx // ch_per_group
    
    # Shared memory for input tile
    shared_input = tl.shared_ptr(input_ptr, (input_depth, input_width, input_height))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kd in range(kernel_depth):
        for kw in range(kernel_width):
            for kh in range(kernel_height):
                # Calculate input positions
                input_d = out_d_idx * stride_d + kd - padding_d
                input_w = out_d_idx * stride_w + kw - padding_w  
                input_h = out_d_idx * stride_h + kh - padding_h
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_w >= 0 and input_w < input_width and
                    input_h >= 0 and input_h < input_height):
                    
                    # Calculate input index
                    input_idx = (
                        batch_idx * (in_channels * input_depth * input_width * input_height) +
                        (out_ch_idx % ch_per_group) * (input_depth * input_width * input_height) +
                        input_d * (input_width * input_height) +
                        input_w * input_height +
                        input_h
                    )
                    
                    # Calculate weight index
                    weight_idx = (
                        out_ch_idx * (in_channels // groups * kernel_depth * kernel_width * kernel_height) +
                        (out_ch_idx % ch_per_group) * (kernel_depth * kernel_width * kernel_height) +
                        kd * (kernel_width * kernel_height) +
                        kw * kernel_height +
                        kh
                    )
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_idx = out_ch_idx
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Store result
    if batch_idx < batch_size and out_ch_idx < out_channels:
        output_idx = (
            batch_idx * (out_channels * output_depth * output_width * output_height) +
            out_ch_idx * (output_depth * output_width * output_height) +
            out_d_idx * (output_width * output_height)
        )
        tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias=None, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1):
    """
    Custom Triton implementation of 3D Transposed Convolution
    """
    # Extract dimensions
    batch_size, in_channels, input_depth, input_width, input_height = input_tensor.shape
    out_channels, _, kernel_depth, kernel_width, kernel_height = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_w, stride_h = stride
    pad_d, pad_w, pad_h = padding
    out_pad_d, out_pad_w, out_pad_h = output_padding
    
    output_depth = (input_depth - 1) * stride_d - 2 * pad_d + kernel_depth + out_pad_d
    output_width = (input_width - 1) * stride_w - 2 * pad_w + kernel_width + out_pad_w
    output_height = (input_height - 1) * stride_h - 2 * pad_h + kernel_height + out_pad_h
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_width, output_height, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
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
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_width,
        input_height,
        output_depth,
        output_width,
        output_height,
        kernel_depth,
        kernel_width,
        kernel_height,
        stride_d,
        stride_w,
        stride_h,
        pad_d,
        pad_w,
        pad_h,
        out_pad_d,
        out_pad_w,
        out_pad_h,
        groups,
        BLOCK_SIZE=1024,
        GROUP_SIZE=32
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use custom Triton kernel
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
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, stride={self.stride}, '
            f'padding={self.padding}, output_padding={self.output_padding}, '
            f'groups={self.groups}, bias={self.bias is not None}'
        )