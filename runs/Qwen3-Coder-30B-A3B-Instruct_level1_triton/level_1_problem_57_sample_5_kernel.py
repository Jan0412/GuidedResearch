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
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    group_id = tl.program_id(2)
    
    # Calculate output dimensions
    out_h = output_height
    out_w = output_width
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over kernel elements
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            # Calculate input coordinates
            input_h_start = kh * stride_h - padding_h
            input_w_start = kw * stride_w - padding_w
            
            # Load input tile
            input_tile = tl.load(input_ptr + 
                               batch_id * in_channels * input_height * input_width +
                               group_id * (in_channels // groups) * input_height * input_width +
                               tl.arange(0, BLOCK_SIZE)[:, None] * input_width +
                               tl.arange(0, BLOCK_SIZE)[None, :] +
                               input_h_start * input_width + input_w_start)
            
            # Load weight tile
            weight_tile = tl.load(weight_ptr + 
                                out_ch_id * groups * in_channels * kernel_height * kernel_width +
                                group_id * (in_channels // groups) * kernel_height * kernel_width +
                                kh * kernel_width + kw)
            
            # Accumulate
            acc += input_tile * weight_tile
    
    # Store output
    tl.store(output_ptr + 
             batch_id * out_channels * out_height * out_width +
             out_ch_id * out_height * out_width +
             tl.arange(0, BLOCK_SIZE)[:, None] * out_width +
             tl.arange(0, BLOCK_SIZE)[None, :], acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output size
    output_height = (input_height - 1) * stride - 2 * padding + kernel_height + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + kernel_width + output_padding
    
    # Prepare output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        groups
    )
    
    BLOCK_SIZE = 16
    GROUP_SIZE = 8
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride,
        stride,
        padding,
        padding,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
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
        
        # Initialize weights using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )