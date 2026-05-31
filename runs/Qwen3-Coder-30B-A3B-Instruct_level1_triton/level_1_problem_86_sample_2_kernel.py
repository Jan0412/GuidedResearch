import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_h_idx = tl.program_id(2)
    
    # Calculate starting position for this block
    start_channel = channel_idx * CHANNELS_PER_BLOCK
    
    # Shared memory for input tile and weight
    TILE_H = output_height + 2 * padding_h
    TILE_W = output_width + 2 * padding_w
    
    # Load weights (kernel)
    weight = tl.load(weight_ptr + channel_idx * kernel_h * kernel_w)
    
    # Process multiple channels if needed
    for c in range(CHANNELS_PER_BLOCK):
        if start_channel + c >= in_channels:
            break
            
        # Load input region for this channel
        input_channel = input_ptr + batch_idx * in_channels * height * width + (start_channel + c) * height * width
        
        # Loop over output width
        for output_w_idx in range(output_width):
            # Calculate input positions
            input_h_start = output_h_idx * stride_h - padding_h
            input_w_start = output_w_idx * stride_w - padding_w
            
            # Compute convolution
            acc = 0.0
            
            # Iterate through kernel
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    ih = input_h_start + kh * dilation_h
                    iw = input_w_start + kw * dilation_w
                    
                    # Check bounds
                    if ih >= 0 and ih < height and iw >= 0 and iw < width:
                        input_val = tl.load(input_channel + ih * width + iw)
                        weight_val = weight + kh * kernel_w + kw
                        acc += input_val * weight_val
            
            # Store output
            output_idx = batch_idx * in_channels * output_height * output_width + \
                         (start_channel + c) * output_height * output_width + \
                         output_h_idx * output_width + output_w_idx
            tl.store(output_ptr + output_idx, acc)

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    hw_idx = tl.program_id(2)
    
    # Calculate output position
    output_offset = batch_idx * out_channels * output_height * output_width + \
                    out_ch_idx * output_height * output_width + hw_idx
    
    # Initialize accumulator
    acc = 0.0
    
    # Load weight for this output channel
    weight_row = weight_ptr + out_ch_idx * in_channels
    
    # Accumulate over input channels
    for c in range(in_channels):
        input_val = tl.load(input_ptr + batch_idx * in_channels * output_height * output_width + 
                           c * output_height * output_width + hw_idx)
        weight_val = tl.load(weight_row + c)
        acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + output_offset, acc)

def triton_depthwise_conv2d(input_tensor, weight, bias, stride, padding, dilation):
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    
    # Calculate output dimensions
    output_height = (height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    output_width = (width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare kernel parameters
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 4
    
    # Grid configuration
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        output_height
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        output_height,
        output_width,
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK
    )
    
    return output

def triton_pointwise_conv2d(input_tensor, weight, bias):
    batch_size, in_channels, output_height, output_width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        output_height * output_width
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        output_height,
        output_width,
        output_height,
        output_width,
        BLOCK_SIZE=128
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernels for both operations
        # Depthwise convolution
        dw_weight = self.depthwise.weight
        dw_bias = self.depthwise.bias
        dw_stride = (self.depthwise.stride,) if isinstance(self.depthwise.stride, int) else self.depthwise.stride
        dw_padding = (self.depthwise.padding,) if isinstance(self.depthwise.padding, int) else self.depthwise.padding
        dw_dilation = (self.depthwise.dilation,) if isinstance(self.depthwise.dilation, int) else self.depthwise.dilation
        
        # Apply Triton depthwise convolution
        x = triton_depthwise_conv2d(x, dw_weight, dw_bias, dw_stride, dw_padding, dw_dilation)
        
        # Pointwise convolution
        pw_weight = self.pointwise.weight
        pw_bias = self.pointwise.bias
        
        # Apply Triton pointwise convolution
        x = triton_pointwise_conv2d(x, pw_weight, pw_bias)
        
        return x