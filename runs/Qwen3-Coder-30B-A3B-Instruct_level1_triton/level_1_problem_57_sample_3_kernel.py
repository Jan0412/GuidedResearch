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
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate output position
    out_h = out_h_idx * BLOCK_SIZE_H
    out_w = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2*padding_h, BLOCK_SIZE_W + 2*padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group offsets
        group_in_ch = in_channels // groups
        group_out_ch = out_channels // groups
        
        # Calculate channel ranges
        in_ch_start = g * group_in_ch
        out_ch_start = g * group_out_ch
        
        # Check if this thread block should process this group
        if out_c_idx >= out_ch_start and out_c_idx < out_ch_start + group_out_ch:
            # Calculate output channel offset
            out_ch_offset = out_c_idx - out_ch_start
            
            # Loop over kernel
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input coordinates
                    input_h = out_h * stride_h - padding_h + kh
                    input_w = out_w * stride_w - padding_w + kw
                    
                    # Check bounds
                    if input_h >= 0 and input_h < input_height and input_w >= 0 and input_w < input_width:
                        # Load input value
                        input_val = tl.load(input_ptr + 
                                          batch_idx * (in_channels * input_height * input_width) +
                                          (in_ch_start + 0) * (input_height * input_width) +
                                          input_h * input_width + input_w)
                        
                        # Load weight value
                        weight_val = tl.load(weight_ptr + 
                                           out_c_idx * (group_in_ch * kernel_height * kernel_width) +
                                           (in_ch_start + 0) * (kernel_height * kernel_width) +
                                           kh * kernel_width + kw)
                        
                        # Accumulate
                        acc += input_val * weight_val
    
    # Store result
    if out_h < output_height and out_w < output_width:
        tl.store(output_ptr + 
                batch_idx * (out_channels * output_height * output_width) +
                out_c_idx * (output_height * output_width) +
                out_h * output_width + out_w,
                acc)

def triton_conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1):
    """
    Triton implementation of ConvTranspose2d
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - 1) * stride - 2 * padding + kernel_height + output_padding
    output_width = (input_width - 1) * stride - 2 * padding + kernel_width + output_padding
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Set up grid configuration
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 8
    
    # Grid dimensions
    grid_batch = batch_size
    grid_out_channels = out_channels
    grid_out_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_out_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    conv_transpose2d_kernel[(grid_batch, grid_out_channels, grid_out_h, grid_out_w)](
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
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        GROUP_SIZE_M=8
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
        
        # Initialize weights and biases
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
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding, 
            groups=self.groups
        )

    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'kernel_size={self.kernel_size}, stride={self.stride}, '
            f'padding={self.padding}, output_padding={self.output_padding}, '
            f'groups={self.groups}, bias={self.bias is not None}'
        )