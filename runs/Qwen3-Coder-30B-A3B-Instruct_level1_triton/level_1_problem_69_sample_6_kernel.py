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
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE_H: tl.constexpr,
    OUTPUT_BLOCK_SIZE_W: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h_start = tl.program_id(2) * OUTPUT_BLOCK_SIZE_H
    out_w_start = tl.program_id(3) * OUTPUT_BLOCK_SIZE_W
    
    # Shared memory for input tile and weight tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(OUTPUT_BLOCK_SIZE_H + 2 * padding_h, OUTPUT_BLOCK_SIZE_W + 2 * padding_w))
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(kernel_height, kernel_width))
    
    # Initialize accumulator
    acc = tl.zeros((OUTPUT_BLOCK_SIZE_H, OUTPUT_BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific pointers
        group_offset_in = g * (in_channels // groups)
        group_offset_out = g * (out_channels // groups)
        
        # Load weight for this group
        weight_base = weight_ptr + group_offset_out * in_channels * kernel_height * kernel_width + \
                      out_c_idx * in_channels * kernel_height * kernel_width
        
        # Loop over kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate output coordinates
                out_h = out_h_start + kh * dilation_h
                out_w = out_w_start + kw * dilation_w
                
                # Check bounds
                if out_h >= 0 and out_h < output_height and out_w >= 0 and out_w < output_width:
                    # Load weight
                    weight_val = tl.load(weight_base + kh * in_channels * kernel_width + kw * in_channels + group_offset_in)
                    
                    # Calculate input coordinates
                    input_h = out_h - padding_h
                    input_w = out_w - padding_w
                    
                    # Load input if within bounds
                    if input_h >= 0 and input_h < input_height and input_w >= 0 and input_w < input_width:
                        input_val = tl.load(input_ptr + batch_idx * in_channels * input_height * input_width + 
                                          group_offset_in * input_height * input_width + 
                                          input_h * input_width + input_w)
                        acc += input_val * weight_val
    
    # Write output
    for i in range(OUTPUT_BLOCK_SIZE_H):
        for j in range(OUTPUT_BLOCK_SIZE_W):
            out_h = out_h_start + i
            out_w = out_w_start + j
            
            if out_h < output_height and out_w < output_width:
                output_idx = batch_idx * out_channels * output_height * output_width + \
                           out_c_idx * output_height * output_width + \
                           out_h * output_width + out_w
                tl.store(output_ptr + output_idx, acc[i, j])

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Custom Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_height - 1) + 1 + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_width - 1) + 1 + output_padding[1]
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    grid = (
        batch_size,
        out_channels,
        math.ceil(output_height / 16),
        math.ceil(output_width / 16)
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
        BLOCK_SIZE=1024,
        OUTPUT_BLOCK_SIZE_H=16,
        OUTPUT_BLOCK_SIZE_W=16
    )
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution operation with asymmetric input and kernel size.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), output_padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
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
        Performs the transposed 2D convolution using Triton kernel.
        """
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )

# Note: The current Triton implementation above is a simplified version and may require further optimization
# for production use, particularly regarding shared memory usage and load balancing.