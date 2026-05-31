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
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_H_PER_BLOCK: tl.constexpr,
    OUTPUT_W_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    out_c_block = tl.program_id(2)
    
    # Calculate output dimensions per block
    output_h_start = tl.program_id(3) * OUTPUT_H_PER_BLOCK
    output_w_start = tl.program_id(4) * OUTPUT_W_PER_BLOCK
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(OUTPUT_H_PER_BLOCK + 2 * padding_h, OUTPUT_W_PER_BLOCK + 2 * padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_H_PER_BLOCK, OUTPUT_W_PER_BLOCK), dtype=tl.float32)
    
    # Loop over channels
    for c in range(0, in_channels, CHANNELS_PER_BLOCK):
        # Load weights for this channel block
        weight_offset = group_idx * (out_channels // groups) * in_channels * kernel_height * kernel_width + \
                       out_c_block * in_channels * kernel_height * kernel_width + \
                       c * kernel_height * kernel_width
        
        # Load input tile (with padding)
        input_offset = batch_idx * in_channels * input_height * input_width + \
                      c * input_height * input_width
        
        # Process each kernel position
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate positions
                input_h = output_h_start * stride_h + kh - padding_h
                input_w = output_w_start * stride_w + kw - padding_w
                
                # Check bounds
                if input_h >= 0 and input_h < input_height and input_w >= 0 and input_w < input_width:
                    # Load input value
                    input_val = tl.load(input_ptr + input_offset + input_h * input_width + input_w, mask=True)
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + weight_offset + kh * kernel_width + kw)
                    
                    # Accumulate
                    acc += input_val * weight_val
    
    # Write output
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_c_block * output_height * output_width + \
                   output_h_start * output_width + output_w_start
    
    # Only write valid locations
    for oh in range(OUTPUT_H_PER_BLOCK):
        for ow in range(OUTPUT_W_PER_BLOCK):
            if output_h_start + oh < output_height and output_w_start + ow < output_width:
                tl.store(output_ptr + output_offset + oh * output_width + ow, acc[oh, ow])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    # Input shape: (batch, in_channels, height, width)
    # Weight shape: (in_channels, out_channels // groups, kernel_h, kernel_w)
    # Output shape: (batch, out_channels, out_height, out_width)
    
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    out_height = (input_height - 1) * stride - 2 * padding + kernel_height + output_padding
    out_width = (input_width - 1) * stride - 2 * padding + kernel_width + output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,  # batch dimension
        groups,      # group dimension
        out_channels // groups,  # output channel dimension
        (out_height + 7) // 8,   # output height dimension
        (out_width + 7) // 8     # output width dimension
    )
    
    # Launch kernel
    conv_transpose2d_kernel[grid](
        input_tensor,
        weight,
        output,
        input_height,
        input_width,
        out_height,
        out_width,
        in_channels,
        out_channels,
        kernel_height,
        kernel_width,
        stride,
        stride,
        padding,
        padding,
        groups,
        BLOCK_SIZE=1024,
        CHANNELS_PER_BLOCK=32,
        OUTPUT_H_PER_BLOCK=8,
        OUTPUT_W_PER_BLOCK=8
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
        if groups == 1:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
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