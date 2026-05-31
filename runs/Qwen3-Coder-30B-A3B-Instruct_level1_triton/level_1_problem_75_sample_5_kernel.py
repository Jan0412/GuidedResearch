import torch
import torch.nn as nn
import torch.nn.functional as F
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
    dilation_h,
    dilation_w,
    groups,
    channels_per_group,
    output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program processes one output element
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_elements
    
    # Calculate which output element this program handles
    output_idx = offsets % (output_height * output_width)
    batch_idx = (offsets // (output_height * output_width)) % batch_size
    channel_idx = (offsets // (output_height * output_width * batch_size)) % out_channels
    
    # Decompose output indices
    out_h = output_idx // output_width
    out_w = output_idx % output_width
    
    # Calculate corresponding input indices
    in_h_start = out_h - kernel_height * dilation_h + dilation_h
    in_w_start = out_w - kernel_width * dilation_w + dilation_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Handle bias if present
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + channel_idx, mask=channel_idx < out_channels)
    
    # Process all input channels and groups
    for g in range(groups):
        group_start = g * channels_per_group
        group_end = group_start + channels_per_group
        
        # Check if this channel belongs to current group
        if channel_idx >= group_start and channel_idx < group_end:
            # Calculate which channel within the group
            channel_in_group = channel_idx - group_start
            
            # Iterate through kernel positions
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input coordinates
                    ih = in_h_start + kh * dilation_h
                    iw = in_w_start + kw * dilation_w
                    
                    # Check if input coordinates are valid
                    if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                        # Calculate input index
                        input_idx = (
                            batch_idx * (in_channels * input_height * input_width) +
                            g * channels_per_group * input_height * input_width +
                            channel_in_group * input_height * input_width +
                            ih * input_width +
                            iw
                        )
                        
                        # Calculate weight index
                        weight_idx = (
                            channel_idx * kernel_height * kernel_width +
                            kh * kernel_width +
                            kw
                        )
                        
                        # Load input value and weight
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Store result
    output_idx_total = (
        batch_idx * (out_channels * output_height * output_width) +
        channel_idx * output_height * output_width +
        out_h * output_width +
        out_w
    )
    
    tl.store(output_ptr + output_idx_total, acc, mask=mask)

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of ConvTranspose2d
    """
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_width - 1) + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare for kernel launch
    channels_per_group = in_channels // groups
    output_elements = batch_size * out_channels * output_height * output_width
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Determine grid size
    grid_size = (output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv_transpose2d_kernel[grid_size](
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
        dilation[0],
        dilation[1],
        groups,
        channels_per_group,
        output_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version using Triton kernel for ConvTranspose2d
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel
        """
        # Use our Triton kernel implementation
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )