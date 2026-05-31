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
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    output_x = tl.program_id(3)
    
    # Calculate global thread index within channel block
    tid = tl.program_id(4)
    
    # Shared memory for input tile and kernel
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    shared_kernel = tl.shared_memory(dtype=tl.float32, shape=(KERNEL_H, KERNEL_W))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel elements
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input coordinates
            ih = output_y * stride_h + kh * dilation_h - padding_h
            iw = output_x * stride_w + kw * dilation_w - padding_w
            
            # Check bounds
            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                # Load input value
                input_val = tl.load(input_ptr + 
                                  batch_idx * (in_channels * height * width) +
                                  channel_idx * (height * width) +
                                  ih * width + iw,
                                  mask=(ih < height) & (iw < width),
                                  other=0.0)
                
                # Load kernel value
                kernel_val = tl.load(weight_ptr + 
                                   channel_idx * (kernel_h * kernel_w) +
                                   kh * kernel_w + kw)
                
                # Accumulate
                acc += input_val * kernel_val
    
    # Store result
    if output_y < output_height and output_x < output_width:
        tl.store(output_ptr + 
                batch_idx * (in_channels * output_height * output_width) +
                channel_idx * (output_height * output_width) +
                output_y * output_width + output_x,
                acc[0])

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
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    output_x = tl.program_id(3)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels
    for in_channel_idx in range(in_channels):
        # Load input value
        input_val = tl.load(input_ptr + 
                           batch_idx * (in_channels * output_height * output_width) +
                           in_channel_idx * (output_height * output_width) +
                           output_y * output_width + output_x)
        
        # Load weight value
        weight_val = tl.load(weight_ptr + 
                            out_channel_idx * in_channels +
                            in_channel_idx)
        
        # Accumulate
        acc += input_val * weight_val
    
    # Store result
    if output_y < output_height and output_x < output_width:
        tl.store(output_ptr + 
                batch_idx * (out_channels * output_height * output_width) +
                out_channel_idx * (output_height * output_width) +
                output_y * output_width + output_x,
                acc[0])

def triton_depthwise_conv2d(input_tensor, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output dimensions
    output_height = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    output_width = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 16
    CHANNELS_PER_BLOCK = 8
    
    # Grid configuration
    grid = (
        batch_size,
        (in_channels + CHANNELS_PER_BLOCK - 1) // CHANNELS_PER_BLOCK,
        output_height,
        output_width,
        min(CHANNELS_PER_BLOCK, in_channels)
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
        output_height,
        output_width,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
    )
    
    return output

def triton_pointwise_conv2d(input_tensor, weight):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 16
    
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
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using Triton kernels.
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
            self.depthwise_bias = nn.Parameter(torch.zeros(in_channels))
            self.pointwise_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('depthwise_bias', None)
            self.register_parameter('pointwise_bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Depthwise convolution with Triton kernel
        x = triton_depthwise_conv2d(
            x, 
            self.depthwise_weight, 
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation)
        )
        
        # Pointwise convolution with Triton kernel
        x = triton_pointwise_conv2d(x, self.pointwise_weight)
        
        # Add biases if present
        if self.bias:
            x = x + self.pointwise_bias.view(1, -1, 1, 1)
            
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.bias}'