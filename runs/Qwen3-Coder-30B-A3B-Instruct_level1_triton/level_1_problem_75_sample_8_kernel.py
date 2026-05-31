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
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    groups,
    channels_per_group,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread IDs
    block_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    # Calculate output elements handled by this block
    output_elements_start = block_id * OUTPUT_ELEMENTS_PER_BLOCK
    output_elements_end = min(output_elements_start + OUTPUT_ELEMENTS_PER_BLOCK, 
                             batch_size * out_channels * output_height * output_width)
    
    # Calculate which output element this block is working on
    if output_elements_end > output_elements_start:
        batch_idx = (output_elements_start // (out_channels * output_height * output_width)) % batch_size
        channel_idx = (output_elements_start // (output_height * output_width)) % out_channels
        out_h = (output_elements_start // output_width) % output_height
        out_w = output_elements_start % output_width
        
        # Calculate which group this channel belongs to
        group_idx = channel_idx // channels_per_group
        
        # Initialize accumulator
        acc = tl.zeros((1,), dtype=tl.float32)
        
        # Handle bias if present
        if bias_ptr is not None:
            acc = tl.load(bias_ptr + channel_idx, mask=True)
        
        # Loop over kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                ih = out_h * stride_height - padding_height + kh * dilation_height
                iw = out_w * stride_width - padding_width + kw * dilation_width
                
                # Check bounds
                if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                    # Calculate input index
                    input_idx = batch_idx * (in_channels * input_height * input_width) + \
                               (group_idx * channels_per_group + (channel_idx % channels_per_group)) * (input_height * input_width) + \
                               ih * input_width + iw
                    
                    # Calculate weight index
                    weight_idx = group_idx * (channels_per_group * kernel_height * kernel_width * out_channels) + \
                                (channel_idx % channels_per_group) * (kernel_height * kernel_width) + \
                                kh * kernel_width + kw
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    
                    # Accumulate
                    acc += input_val * weight_val
        
        # Store result
        output_idx = batch_idx * (out_channels * output_height * output_width) + \
                    channel_idx * (output_height * output_width) + \
                    out_h * output_width + out_w
        
        tl.store(output_ptr + output_idx, acc, mask=True)

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Triton implementation of ConvTranspose2d using custom kernel
    """
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_height - 1) + 1
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_width - 1) + 1
    
    # Ensure tensors are contiguous and on CUDA
    input_tensor = input_tensor.contiguous().cuda()
    weight = weight.contiguous().cuda()
    if bias is not None:
        bias = bias.contiguous().cuda()
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device='cuda', dtype=torch.float32)
    
    # Constants
    channels_per_group = in_channels // groups
    OUTPUT_ELEMENTS_PER_BLOCK = 256
    BLOCK_SIZE = 128
    GROUPS_PER_BLOCK = 1
    
    # Grid configuration
    grid = (
        (batch_size * out_channels * output_height * output_width + OUTPUT_ELEMENTS_PER_BLOCK - 1) // OUTPUT_ELEMENTS_PER_BLOCK,
        groups
    )
    
    # Launch kernel
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
        dilation[0],
        dilation[1],
        groups,
        channels_per_group,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
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
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]