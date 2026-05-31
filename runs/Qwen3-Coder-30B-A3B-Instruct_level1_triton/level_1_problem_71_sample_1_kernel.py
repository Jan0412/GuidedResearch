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
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    output_padding_h,
    output_padding_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate group information
    group_size = out_channels // groups
    group_idx = out_ch_idx // group_size
    
    # Shared memory for weight tiles
    shared_weight = tl.shared_memory(dtype=tl.float32, shape=(GROUPS_PER_BLOCK, kernel_h, kernel_w))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input position
            input_h = out_h_idx * stride_h + kh - padding_h
            input_w = out_w_idx * stride_w + kw - padding_w
            
            # Check if input position is valid
            if input_h >= 0 and input_h < height_in and input_w >= 0 and input_w < width_in:
                # Load weight
                weight_val = tl.load(weight_ptr + 
                                   (group_idx * group_size + out_ch_idx % group_size) * in_channels * kernel_h * kernel_w +
                                   kh * kernel_w + kw)
                
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * in_channels * height_in * width_in +
                                  (out_ch_idx % group_size) * height_in * width_in +
                                  input_h * width_in + input_w)
                
                acc += weight_val * input_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    # Store result
    output_idx = batch_idx * out_channels * height_out * width_out + \
                 out_ch_idx * height_out * width_out + \
                 out_h_idx * width_out + out_w_idx
    tl.store(output_ptr + output_idx, acc[0])

def triton_conv_transpose2d(input_tensor, weight, bias, stride, padding, output_padding, groups):
    """
    Triton implementation of ConvTranspose2d
    """
    # Get dimensions
    batch_size, in_channels, height_in, width_in = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    height_out = (height_in - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    width_out = (width_in - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height_out, width_out, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare kernel launch parameters
    BLOCK_SIZE = 1024
    GROUPS_PER_BLOCK = 4
    
    # Grid dimensions
    grid = (
        batch_size,
        out_channels,
        height_out,
        width_out
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
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        output_padding[0],
        output_padding[1],
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
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
            self.stride, 
            self.padding, 
            self.output_padding, 
            self.groups
        )