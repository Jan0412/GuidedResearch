import torch
import torch.nn as nn
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
    output_h,
    output_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    output_x = tl.program_id(3)
    
    # Calculate tile boundaries
    tile_start_y = output_y * TILE_H
    tile_start_x = output_x * TILE_W
    
    # Shared memory for input tile and kernel
    input_tile = tl.shared_tensor(tl.float32, (TILE_H + 2 * padding_h, TILE_W + 2 * padding_w))
    kernel_tile = tl.shared_tensor(tl.float32, (kernel_h, kernel_w))
    
    # Initialize accumulator
    acc = tl.zeros((TILE_H, TILE_W), dtype=tl.float32)
    
    # Loop over kernel elements
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            # Load kernel element
            k_val = tl.load(weight_ptr + channel_idx * kernel_h * kernel_w + ky * kernel_w + kx)
            
            # Load input tile with proper boundary handling
            input_y = tile_start_y * stride_h + ky * dilation_h - padding_h
            input_x = tile_start_x * stride_w + kx * dilation_w - padding_w
            
            # Load input elements for this tile
            for ty in range(TILE_H):
                for tx in range(TILE_W):
                    in_y = input_y + ty
                    in_x = input_x + tx
                    
                    # Boundary check
                    if in_y >= 0 and in_y < height and in_x >= 0 and in_x < width:
                        input_val = tl.load(input_ptr + 
                                          batch_idx * in_channels * height * width +
                                          channel_idx * height * width +
                                          in_y * width + in_x)
                    else:
                        input_val = 0.0
                    
                    input_tile[ty][tx] = input_val
            
            # Compute partial products
            for ty in range(TILE_H):
                for tx in range(TILE_W):
                    acc[ty][tx] += input_tile[ty][tx] * k_val
    
    # Store results
    for ty in range(TILE_H):
        for tx in range(TILE_W):
            out_y = tile_start_y + ty
            out_x = tile_start_x + tx
            
            if out_y < output_h and out_x < output_w:
                tl.store(output_ptr + 
                        batch_idx * in_channels * output_h * output_w +
                        channel_idx * output_h * output_w +
                        out_y * output_w + out_x,
                        acc[ty][tx])

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
    output_h,
    output_w,
    BLOCK_SIZE: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    output_x = tl.program_id(3)
    
    # Calculate output position
    output_pos = batch_idx * out_channels * output_h * output_w + \
                 channel_idx * output_h * output_w + \
                 output_y * output_w + output_x
    
    # Accumulate over input channels
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Load input for this position
    for c in range(in_channels):
        input_val = tl.load(input_ptr + 
                          batch_idx * in_channels * output_h * output_w +
                          c * output_h * output_w +
                          output_y * output_w + output_x)
        
        weight_val = tl.load(weight_ptr + 
                           c * out_channels + channel_idx)
        
        acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + output_pos, acc[0])

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1)):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    output_h = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    output_w = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    # Allocate output
    output = torch.empty(batch_size, in_channels, output_h, output_w, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    TILE_H = 8
    TILE_W = 8
    CHANNELS_PER_BLOCK = 1
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (output_h + TILE_H - 1) // TILE_H,
        (output_w + TILE_W - 1) // TILE_W
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
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        output_h,
        output_w,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        TILE_H=TILE_H,
        TILE_W=TILE_W
    )
    
    return output

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Triton implementation of pointwise convolution
    """
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Allocate output
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        height,
        width
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using Triton kernels.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Initialize weights
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Apply depthwise convolution with Triton kernel
        x = triton_depthwise_conv2d(
            x, 
            self.depthwise_weight, 
            None,  # No bias for depthwise conv
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation)
        )
        
        # Apply pointwise convolution with Triton kernel
        x = triton_pointwise_conv2d(
            x,
            self.pointwise_weight,
            self.bias_param
        )
        
        return x