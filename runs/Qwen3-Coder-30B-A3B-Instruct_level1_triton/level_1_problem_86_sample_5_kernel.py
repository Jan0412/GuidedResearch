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
    bias_ptr,
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
    out_height,
    out_width,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    channel_block = tl.program_id(3)
    
    # Calculate output indices
    out_h_start = out_h_idx * stride_h
    out_w_start = out_w_idx * stride_w
    
    # Calculate input indices with padding
    in_h_start = out_h_start - padding_h
    in_w_start = out_w_start - padding_w
    
    # Shared memory for input tile
    shared_input = tl.shared_tensor(tl.float32, (BLOCK_SIZE, BLOCK_SIZE))
    
    # Process channels in chunks
    channel_start = channel_block * CHANNELS_PER_BLOCK
    channel_end = min(channel_start + CHANNELS_PER_BLOCK, in_channels)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input positions
            in_h = in_h_start + kh * dilation_h
            in_w = in_w_start + kw * dilation_w
            
            # Check bounds
            if in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * (in_channels * height * width) +
                                  channel_start * (height * width) +
                                  in_h * width + in_w,
                                  mask=(in_h < height) & (in_w < width),
                                  other=0.0)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + 
                                   channel_start * kernel_h * kernel_w +
                                   kh * kernel_w + kw,
                                   mask=(channel_start < in_channels),
                                   other=0.0)
                
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel_start, mask=(channel_start < in_channels), other=0.0)
        acc += bias_val
    
    # Store output
    if channel_start < in_channels:
        output_idx = batch_idx * (in_channels * out_height * out_width) + \
                    channel_start * (out_height * out_width) + \
                    out_h_idx * out_width + out_w_idx
        tl.store(output_ptr + output_idx, acc, mask=(channel_start < in_channels))

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    out_height,
    out_width,
    BLOCK_SIZE: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    out_channel_idx = tl.program_id(3)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over input channels
    for c in range(in_channels):
        # Load input value
        input_val = tl.load(input_ptr + 
                          batch_idx * (in_channels * out_height * out_width) +
                          c * (out_height * out_width) +
                          out_h_idx * out_width + out_w_idx,
                          other=0.0)
        
        # Load weight value
        weight_val = tl.load(weight_ptr + 
                           out_channel_idx * in_channels + c,
                           other=0.0)
        
        acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx, other=0.0)
        acc += bias_val
    
    # Store output
    output_idx = batch_idx * (out_channels * out_height * out_width) + \
                out_channel_idx * (out_height * out_width) + \
                out_h_idx * out_width + out_w_idx
    tl.store(output_ptr + output_idx, acc)

def triton_depthwise_conv2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1)
):
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_height = (height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) // stride[0] + 1
    out_width = (width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) // stride[1] + 1
    
    # Prepare output tensor
    output = torch.empty(batch_size, in_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Configure grid
    grid = (
        batch_size,
        out_height,
        out_width,
        (in_channels + 31) // 32  # Channel blocks
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
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
        out_height,
        out_width,
        BLOCK_SIZE=128,
        CHANNELS_PER_BLOCK=32
    )
    
    return output

def triton_pointwise_conv2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None
):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    out_height, out_width = height, width  # Same spatial dimensions
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Configure grid
    grid = (
        batch_size,
        out_height,
        out_width,
        out_channels
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        out_height,
        out_width,
        BLOCK_SIZE=128
    )
    
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
        
        # Create convolution weights
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.depthwise_bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('depthwise_bias', None)
            
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        if bias:
            self.pointwise_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('pointwise_bias', None)
        
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
            self.depthwise_bias,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation)
        )
        
        # Apply pointwise convolution with Triton kernel
        x = triton_pointwise_conv2d(
            x,
            self.pointwise_weight,
            self.pointwise_bias
        )
        
        return x