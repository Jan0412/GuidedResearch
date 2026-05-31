import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_height_stride,
    weight_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    bias_channel_stride,
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
    group_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    
    # Calculate output width index
    out_w_idx = tl.program_id(3)
    
    # Shared memory for input tile
    input_tile = tl.shared_ptr(input_ptr, (BLOCK_SIZE_H + 2 * padding_h, BLOCK_SIZE_W + 2 * padding_w))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Calculate group information
    group_id = out_ch_idx // (out_channels // groups)
    group_offset = group_id * group_size
    
    # Process kernel
    for kh in range(0, kernel_height):
        for kw in range(0, kernel_width):
            # Calculate input position
            ih = out_h_idx * stride_h + kh * dilation_h - padding_h
            iw = out_w_idx * stride_w + kw * dilation_w - padding_w
            
            # Load weights
            w = tl.load(weight_ptr + 
                       group_offset * weight_out_channel_stride +
                       out_ch_idx * weight_in_channel_stride +
                       kh * weight_height_stride +
                       kw * weight_width_stride)
            
            # Load input if within bounds
            if ih >= 0 and ih < input_height and iw >= 0 and iw < input_width:
                input_val = tl.load(input_ptr + 
                                  batch_idx * input_batch_stride +
                                  (out_ch_idx % group_size) * input_channel_stride +
                                  ih * input_height_stride +
                                  iw * input_width_stride)
                acc += input_val * w
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_ch_idx * bias_channel_stride)
        acc += bias
    
    # Store output
    if out_h_idx < output_height and out_w_idx < output_width:
        tl.store(output_ptr + 
                batch_idx * output_batch_stride +
                out_ch_idx * output_channel_stride +
                out_h_idx * output_height_stride +
                out_w_idx * output_width_stride,
                acc)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using custom Triton kernel.
        """
        batch_size, in_channels, input_height, input_width = x.shape
        out_channels, _, kernel_height, kernel_width = self.weight.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - (self.dilation * (kernel_height - 1) + 1)) // self.stride + 1
        output_width = (input_width + 2 * self.padding - (self.dilation * (kernel_width - 1) + 1)) // self.stride + 1
        
        # Prepare output tensor
        output = torch.empty(batch_size, out_channels, output_height, output_width, device=x.device, dtype=torch.float32)
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        if self.bias is not None:
            bias = self.bias.contiguous()
        else:
            bias = None
            
        # Define strides for memory access
        batch_stride = x.stride(0)
        channel_stride = x.stride(1)
        height_stride = x.stride(2)
        width_stride = x.stride(3)
        
        weight_out_stride = weight.stride(0)
        weight_in_stride = weight.stride(1)
        weight_h_stride = weight.stride(2)
        weight_w_stride = weight.stride(3)
        
        output_batch_stride = output.stride(0)
        output_channel_stride = output.stride(1)
        output_h_stride = output.stride(2)
        output_w_stride = output.stride(3)
        
        bias_channel_stride = 1 if bias is not None else 0
        
        # Launch kernel
        grid = (
            batch_size,
            out_channels,
            output_height,
            output_width
        )
        
        # Use appropriate block sizes
        BLOCK_SIZE_H = min(16, output_height)
        BLOCK_SIZE_W = min(16, output_width)
        BLOCK_SIZE_C = min(16, out_channels)
        
        # For simplicity, use a fixed group size
        group_size = self.in_channels // self.groups
        
        # Launch kernel
        conv2d_kernel[grid](
            x,
            weight,
            output,
            bias,
            batch_stride,
            channel_stride,
            height_stride,
            width_stride,
            weight_out_stride,
            weight_in_stride,
            weight_h_stride,
            weight_w_stride,
            output_batch_stride,
            output_channel_stride,
            output_h_stride,
            output_w_stride,
            bias_channel_stride,
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.groups,
            group_size,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            GROUP_SIZE=1
        )
        
        return output