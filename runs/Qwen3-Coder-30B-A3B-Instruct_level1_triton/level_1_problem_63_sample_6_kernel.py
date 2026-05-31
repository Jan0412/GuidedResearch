import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    padding_h,
    padding_w,
    stride_h,
    stride_w,
    dilation_h,
    dilation_w,
    has_bias,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr
):
    # Get the block ID for this thread
    pid = tl.program_id(0)
    
    # Calculate which batch/channel this block processes
    batch_id = pid // (out_channels * (output_height // TILE_H) * (output_width // TILE_W))
    remaining = pid % (out_channels * (output_height // TILE_H) * (output_width // TILE_W))
    channel_id = remaining // ((output_height // TILE_H) * (output_width // TILE_W))
    remaining = remaining % ((output_height // TILE_H) * (output_width // TILE_W))
    tile_y = remaining // (output_width // TILE_W)
    tile_x = remaining % (output_width // TILE_W)
    
    # Check bounds
    if batch_id >= batch_size or channel_id >= out_channels:
        return
        
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Process all input channels
    for ic in range(in_channels):
        # Loop over kernel elements
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input position
                ih = tile_y * TILE_H + kh * dilation_h - padding_h
                iw = tile_x * TILE_W + kw * dilation_w - padding_w
                
                # Skip if out of bounds
                if ih < 0 or ih >= input_height or iw < 0 or iw >= input_width:
                    continue
                    
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_id * input_stride_0 +
                                  ic * input_stride_1 +
                                  ih * input_stride_2 +
                                  iw * input_stride_3,
                                  mask=(ih < input_height) & (iw < input_width),
                                  other=0.0)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   channel_id * weight_stride_0 +
                                   ic * weight_stride_1 +
                                   kh * weight_stride_2 +
                                   kw * weight_stride_3)
                
                # Accumulate
                acc += input_val * weight_val
    
    # Apply bias if needed
    if has_bias:
        bias_val = tl.load(bias_ptr + channel_id)
        acc += bias_val
    
    # Store output
    for i in range(TILE_H):
        for j in range(TILE_W):
            if tile_y * TILE_H + i < output_height and tile_x * TILE_W + j < output_width:
                tl.store(output_ptr + 
                        batch_id * output_stride_0 +
                        channel_id * output_stride_1 +
                        (tile_y * TILE_H + i) * output_stride_2 +
                        (tile_x * TILE_W + j) * output_stride_3,
                        acc[i, j])

def triton_conv2d(input_tensor, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton implementation of 2D convolution
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Set up parameters for kernel launch
    BLOCK_SIZE = 1024
    TILE_H = 8
    TILE_W = 8
    
    # Grid size calculation
    total_blocks = batch_size * out_channels * (output_height // TILE_H) * (output_width // TILE_W)
    grid = (total_blocks,)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        padding[0],
        padding[1],
        stride[0],
        stride[1],
        dilation[0],
        dilation[1],
        bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=8,
        TILE_H=TILE_H,
        TILE_W=TILE_W
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized 2D convolution using Triton kernels
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel
        """
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )