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
    GROUPS_PER_BLOCK: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
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
    
    # Shared memory for input tiles
    shared_input = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE * CHANNELS_PER_BLOCK)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Handle bias
    if bias_ptr != 0:
        bias_offset = group_idx * (out_channels // groups)
        acc = tl.load(bias_ptr + bias_offset, mask=(group_idx < groups))
    
    # Loop over kernel dimensions
    for k_d in range(kernel_depth):
        for k_h in range(kernel_height):
            for k_w in range(kernel_width):
                # Calculate input position
                input_d = out_d * stride_d - padding_d + k_d * dilation_d
                input_h = out_h * stride_h - padding_h + k_h * dilation_h
                input_w = out_w * stride_w - padding_w + k_w * dilation_w
                
                # Check bounds
                if (input_d >= 0 and input_d < input_depth and
                    input_h >= 0 and input_h < input_height and
                    input_w >= 0 and input_w < input_width):
                    
                    # Calculate input index
                    input_idx = batch_idx * in_channels * input_depth * input_height * input_width + \
                               group_idx * (in_channels // groups) * input_depth * input_height * input_width + \
                               input_d * input_height * input_width + \
                               input_h * input_width + \
                               input_w
                    
                    # Calculate weight index
                    weight_idx = group_idx * (out_channels // groups) * in_channels * kernel_depth * kernel_height * kernel_width + \
                                (out_d * stride_d - padding_d + k_d * dilation_d) * (in_channels // groups) * kernel_height * kernel_width + \
                                (out_h * stride_h - padding_h + k_h * dilation_h) * (in_channels // groups) * kernel_width + \
                                (out_w * stride_w - padding_w + k_w * dilation_w) * (in_channels // groups)
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + out_pos, acc, mask=True)

def triton_conv_transpose3d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of ConvTranspose3d
    """
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_depth = (input_depth - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_depth - 1) + 1
    output_height = (input_height - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_width - 1) + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    grid = (
        batch_size,
        groups,
        output_depth,
        output_height,
        output_width
    )
    
    # Define block sizes
    BLOCK_SIZE = 1024
    GROUPS_PER_BLOCK = 1
    CHANNELS_PER_BLOCK = 1
    
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
        dilation[0],
        dilation[1],
        dilation[2],
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution using Triton kernel.
        """
        # Convert to contiguous for better performance
        x = x.contiguous()
        
        # Use our Triton implementation
        return triton_conv_transpose3d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )