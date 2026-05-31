import torch
import torch.nn as nn
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
    groups,
    bias_enabled,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Calculate group info
    ch_per_group = out_channels // groups
    group_idx = out_ch_idx // ch_per_group
    
    # Shared memory for input tile
    tile_size = BLOCK_SIZE
    tile_input = tl.shared_memory(dtype=tl.float32, shape=(tile_size, tile_size))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel dimensions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input coordinates
            input_y = out_y * stride_h + kh - padding_h
            input_x = out_x * stride_w + kw - padding_w
            
            # Check bounds
            if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * in_channels * input_height * input_width +
                                  group_idx * (in_channels // groups) * input_height * input_width +
                                  input_y * input_width + input_x)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   out_ch_idx * kernel_height * kernel_width * (in_channels // groups) +
                                   kh * kernel_width * (in_channels // groups) +
                                   kw * (in_channels // groups) +
                                   (out_ch_idx % ch_per_group))
                
                acc += input_val * weight_val
    
    # Add bias if enabled
    if bias_enabled:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Store result
    if out_y < output_height and out_x < output_width:
        tl.store(output_ptr + 
                batch_idx * out_channels * output_height * output_width +
                out_ch_idx * output_height * output_width +
                out_y * output_width + out_x, 
                acc[0])

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Custom Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_height + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_width + output_padding[1]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    grid = (
        batch_size,
        out_channels,
        math.ceil(output_height / 16),
        math.ceil(output_width / 16)
    )
    
    # Launch kernel
    BLOCK_SIZE = 16
    GROUP_SIZE = 16
    
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
        groups,
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
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
            self.output_padding, 
            self.groups
        )