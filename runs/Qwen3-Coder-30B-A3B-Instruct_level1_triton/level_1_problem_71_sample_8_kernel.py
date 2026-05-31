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
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_y = tl.program_id(2)
    out_x = tl.program_id(3)
    
    # Calculate group information
    group_idx = out_ch_idx // CHANNELS_PER_GROUP
    local_ch_idx = out_ch_idx % CHANNELS_PER_GROUP
    
    # Calculate output position
    out_pos = batch_idx * out_channels * output_height * output_width + \
              out_ch_idx * output_height * output_width + \
              out_y * output_width + out_x
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input coordinates
            # For transposed conv, we need to map output coordinates back to input
            # input_y = out_y * stride_h + kh - padding_h
            # input_x = out_x * stride_w + kw - padding_w
            
            input_y = out_y * stride_h + kh - padding_h
            input_x = out_x * stride_w + kw - padding_w
            
            # Check bounds
            if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                # Calculate input position
                input_pos = batch_idx * in_channels * input_height * input_width + \
                           (group_idx * CHANNELS_PER_GROUP + local_ch_idx) * input_height * input_width + \
                           input_y * input_width + input_x
                
                # Calculate weight position
                weight_pos = group_idx * CHANNELS_PER_GROUP * kernel_height * kernel_width + \
                            local_ch_idx * kernel_height * kernel_width + \
                            kh * kernel_width + kw
                
                # Load values
                input_val = tl.load(input_ptr + input_pos, mask=True)
                weight_val = tl.load(weight_ptr + weight_pos, mask=True)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + out_pos, acc[0])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    output_padding_h, output_padding_w = output_padding
    
    output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h
    output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Calculate grid dimensions
    grid = (
        batch_size,
        out_channels,
        output_height,
        output_width
    )
    
    # Constants for kernel
    CHANNELS_PER_GROUP = in_channels // groups
    BLOCK_SIZE = 128
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_height,
        input_width,
        output_height,
        output_width,
        in_channels,
        out_channels,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=groups,
        CHANNELS_PER_GROUP=CHANNELS_PER_GROUP
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            output_padding=(self.output_padding, self.output_padding),
            groups=self.groups
        )

# For compatibility with original interface
def get_inputs():
    batch_size = 8
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    height_in = 512
    width_in = 1024
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [32, 32, 3]  # in_channels, out_channels, kernel_size