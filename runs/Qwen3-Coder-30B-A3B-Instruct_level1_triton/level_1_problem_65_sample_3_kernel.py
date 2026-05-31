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
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    groups,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Calculate output position
    out_h_start = out_h_idx * BLOCK_SIZE_H
    out_w_start = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_tile(input_ptr, [BLOCK_SIZE_H, BLOCK_SIZE_W], [1, 1])
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific pointers
        weight_group_ptr = weight_ptr + g * (out_channels // groups) * in_channels * kernel_h * kernel_w
        output_group_ptr = output_ptr + batch_idx * out_channels * output_height * output_width + g * (out_channels // groups) * output_height * output_width
        
        # Loop over kernel dimensions
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate input positions
                input_h_start = out_h_start * stride_h - padding_h + kh
                input_w_start = out_w_start * stride_w - padding_w + kw
                
                # Check bounds
                if input_h_start >= 0 and input_h_start < input_height and input_w_start >= 0 and input_w_start < input_width:
                    # Load input data
                    input_val = tl.load(input_ptr + batch_idx * in_channels * input_height * input_width + 
                                      g * (in_channels // groups) * input_height * input_width +
                                      input_h_start * input_width + input_w_start)
                    
                    # Load weight
                    weight_val = tl.load(weight_group_ptr + kh * kernel_w * (out_channels // groups) * in_channels + 
                                       kw * (out_channels // groups) * in_channels + 
                                       (out_h_start * stride_h - padding_h + kh) * (out_channels // groups) * in_channels + 
                                       (out_w_start * stride_w - padding_w + kw) * (out_channels // groups) * in_channels)
                    
                    # Accumulate
                    output_val = input_val * weight_val
                    tl.atomic_add(output_group_ptr + out_h_start * output_width + out_w_start, output_val)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1):
    """
    Triton implementation of ConvTranspose2d operation
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    
    # Initialize output tensor
    output = torch.zeros(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    
    # Grid configuration
    grid = (
        batch_size,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
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
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        groups,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        BLOCK_SIZE_C
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
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
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )