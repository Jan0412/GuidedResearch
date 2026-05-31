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
    kernel_size,
    stride,
    padding,
    dilation,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    HEIGHT_PER_BLOCK: tl.constexpr,
    WIDTH_PER_BLOCK: tl.constexpr
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    width_idx = tl.program_id(3)
    
    # Calculate global indices
    out_h_start = height_idx * HEIGHT_PER_BLOCK
    out_w_start = width_idx * WIDTH_PER_BLOCK
    
    # Shared memory for input tile
    input_tile = tl.shared_tensor(tl.float32, (HEIGHT_PER_BLOCK + 2 * padding, WIDTH_PER_BLOCK + 2 * padding))
    
    # Load weights (shared across threads in block)
    weight = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size)
    
    # Loop over kernel elements
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Calculate input position
            in_h = out_h_start * stride + k_h * dilation - padding
            in_w = out_w_start * stride + k_w * dilation - padding
            
            # Load input data
            if in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                input_val = tl.load(input_ptr + 
                                  batch_idx * (in_channels * height * width) +
                                  channel_idx * (height * width) +
                                  in_h * width + in_w)
            else:
                input_val = 0.0
                
            # Store in shared memory
            shared_h = k_h
            shared_w = k_w
            if shared_h < HEIGHT_PER_BLOCK + 2 * padding and shared_w < WIDTH_PER_BLOCK + 2 * padding:
                input_tile[shared_h, shared_w] = input_val
    
    # Compute convolution
    acc = 0.0
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            # Read from shared memory
            shared_h = k_h
            shared_w = k_w
            if shared_h < HEIGHT_PER_BLOCK and shared_w < WIDTH_PER_BLOCK:
                input_val = input_tile[shared_h, shared_w]
                weight_val = weight[k_h * kernel_size + k_w]
                acc += input_val * weight_val
    
    # Write output
    if out_h_start < output_height and out_w_start < output_width:
        out_idx = batch_idx * (in_channels * output_height * output_width) + \
                  channel_idx * (output_height * output_width) + \
                  out_h_start * output_width + out_w_start
        tl.store(output_ptr + out_idx, acc)

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    hw_idx = tl.program_id(2)
    
    # Calculate global indices
    out_c_start = channel_idx * BLOCK_SIZE
    
    # Shared memory for input tile
    input_tile = tl.shared_tensor(tl.float32, (BLOCK_SIZE,))
    
    # Load input data
    input_val = tl.load(input_ptr + 
                      batch_idx * (in_channels * height * width) +
                      channel_idx * (height * width) +
                      hw_idx)
    
    # Compute pointwise convolution
    for c in range(out_channels):
        weight_val = tl.load(weight_ptr + channel_idx * out_channels + c)
        bias_val = tl.load(bias_ptr + c)
        output_val = input_val * weight_val + bias_val
        
        # Store output
        out_idx = batch_idx * (out_channels * height * width) + \
                  c * (height * width) + \
                  hw_idx
        tl.store(output_ptr + out_idx, output_val)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1):
    """Triton implementation of depthwise convolution"""
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_size = weight.shape[-1]
    output_height = (height + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    output_width = (width + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (output_height + 7) // 8,
        (output_width + 7) // 8
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
        kernel_size,
        stride,
        padding,
        dilation,
        output_height,
        output_width,
        BLOCK_SIZE=1024,
        CHANNELS_PER_BLOCK=1,
        HEIGHT_PER_BLOCK=8,
        WIDTH_PER_BLOCK=8
    )
    
    return output

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """Triton implementation of pointwise convolution"""
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        height * width
    )
    
    # Launch kernel
    pointwise_conv2d_kernel[grid](
        input_tensor,
        weight,
        bias,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE=1024
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
        
        # Initialize weights and biases
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_param', None)
            
        # Initialize weights
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Depthwise convolution with Triton kernel
        depthwise_output = triton_depthwise_conv2d(
            x, 
            self.depthwise_weight, 
            None,  # No bias for depthwise
            self.stride, 
            self.padding, 
            self.dilation
        )
        
        # Pointwise convolution with Triton kernel
        pointwise_output = triton_pointwise_conv2d(
            depthwise_output,
            self.pointwise_weight,
            self.bias_param
        )
        
        return pointwise_output

    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias}'