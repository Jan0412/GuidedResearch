import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    
    # Calculate output indices
    output_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total output elements per batch
    total_output_elements = output_depth * output_height * output_width
    
    # Early exit if output index exceeds total elements
    if tl.any(output_idx >= total_output_elements):
        return
    
    # Calculate output coordinates from linear index
    out_d = output_idx // (output_height * output_width)
    out_h = (output_idx % (output_height * output_width)) // output_width
    out_w = output_idx % output_width
    
    # Check bounds
    if out_d >= output_depth or out_h >= output_height or out_w >= output_width:
        return
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for ic in range(in_channels):
        for kd in range(kernel_depth):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input positions
                    input_d = out_d * stride_d - padding_d + kd * dilation_d
                    input_h = out_h * stride_h - padding_h + kh * dilation_h
                    input_w = out_w * stride_w - padding_w + kw * dilation_w
                    
                    # Check if input position is valid
                    if (input_d >= 0 and input_d < input_depth and 
                        input_h >= 0 and input_h < input_height and 
                        input_w >= 0 and input_w < input_width):
                        
                        # Calculate input and weight indices
                        input_idx = (batch_id * in_channels * input_depth * input_height * input_width + 
                                   ic * input_depth * input_height * input_width + 
                                   input_d * input_height * input_width + 
                                   input_h * input_width + 
                                   input_w)
                        
                        weight_idx = (out_ch_id * in_channels * kernel_depth * kernel_height * kernel_width + 
                                    ic * kernel_depth * kernel_height * kernel_width + 
                                    kd * kernel_height * kernel_width + 
                                    kh * kernel_width + 
                                    kw)
                        
                        # Load values and accumulate
                        input_val = tl.load(input_ptr + input_idx, mask=True)
                        weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                        acc += input_val * weight_val
    
    # Store result
    output_idx_total = (batch_id * out_channels * output_depth * output_height * output_width + 
                       out_ch_id * output_depth * output_height * output_width + 
                       out_d * output_height * output_width + 
                       out_h * output_width + 
                       out_w)
    
    tl.store(output_ptr + output_idx_total, acc[0], mask=True)

def triton_conv3d(input_tensor, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1)):
    """
    Triton implementation of 3D convolution
    """
    # Ensure inputs are on CUDA and contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, input_depth, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation
    
    output_depth = (input_depth + 2 * pad_d - (dil_d * (kernel_depth - 1) + 1)) // stride_d + 1
    output_height = (input_height + 2 * pad_h - (dil_h * (kernel_height - 1) + 1)) // stride_h + 1
    output_width = (input_width + 2 * pad_w - (dil_w * (kernel_width - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block size and group size
    BLOCK_SIZE = 256
    GROUP_SIZE = 8
    
    # Launch kernel
    grid = (
        batch_size,
        out_channels,
        (output_depth * output_height * output_width + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    
    # Call the kernel
    conv3d_kernel[grid](
        input_tensor,
        weight,
        output,
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
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE
    )
    
    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution using Triton kernel
        """
        return triton_conv3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_channels={self.in_channels}',
            f'out_channels={self.out_channels}',
            f'kernel_size={self.kernel_size}',
            f'stride={self.stride}',
            f'padding={self.padding}',
            f'dilation={self.dilation}',
            f'groups={self.groups}',
            f'bias={self.bias is not None}'
        ])