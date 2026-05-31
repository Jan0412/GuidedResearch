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
    out_h,
    out_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    channel_block = tl.program_id(3)
    
    # Calculate global indices
    out_h_start = out_h_idx * stride_h
    out_w_start = out_w_idx * stride_w
    
    # Shared memory for input tile
    input_tile = tl.shared_memory(shape=(2 * kernel_h * dilation_h + 1, 2 * kernel_w * dilation_w + 1), dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over kernel elements
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input positions
            ih = out_h_start + kh * dilation_h - padding_h
            iw = out_w_start + kw * dilation_w - padding_w
            
            # Check bounds
            if ih >= 0 and ih < height and iw >= 0 and iw < width:
                # Load input value
                input_val = tl.load(input_ptr + batch_idx * (in_channels * height * width) + 
                                  channel_block * (height * width) + 
                                  ih * width + iw, mask=True)
                
                # Load weight value
                weight_val = tl.load(weight_ptr + channel_block * kernel_h * kernel_w + 
                                   kh * kernel_w + kw, mask=True)
                
                acc += input_val * weight_val
            else:
                # Out of bounds, treat as zero
                pass
    
    # Store result
    if out_h_idx < out_h and out_w_idx < out_w:
        tl.store(output_ptr + batch_idx * (in_channels * out_h * out_w) + 
                channel_block * (out_h * out_w) + 
                out_h_idx * out_w + out_w_idx, acc)

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
    BLOCK_SIZE: tl.constexpr
):
    # Get block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels
    for c in range(in_channels):
        input_val = tl.load(input_ptr + batch_idx * (in_channels * height * width) + 
                          c * (height * width) + 
                          h_idx * width + w_idx, mask=True)
        
        weight_val = tl.load(weight_ptr + channel_idx * in_channels + c, mask=True)
        
        acc += input_val * weight_val
    
    # Store result
    if channel_idx < out_channels:
        tl.store(output_ptr + batch_idx * (out_channels * height * width) + 
                channel_idx * (height * width) + 
                h_idx * width + w_idx, acc)

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0, dilation=1):
    """
    Triton implementation of depthwise convolution
    """
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - (dilation * (kernel_h - 1) + 1)) // stride + 1
    out_w = (width + 2 * padding - (dilation * (kernel_w - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, out_h, out_w, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE = 1024
    CHANNELS_PER_BLOCK = 16
    
    # Grid configuration
    grid = (
        batch_size,
        math.ceil(out_h / stride),
        math.ceil(out_w / stride),
        math.ceil(in_channels / CHANNELS_PER_BLOCK)
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
        out_h,
        out_w,
        stride,
        stride,
        padding,
        padding,
        dilation,
        dilation,
        BLOCK_SIZE,
        CHANNELS_PER_BLOCK
    )
    
    return output

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Triton implementation of pointwise convolution
    """
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Define block sizes
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
        BLOCK_SIZE
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using Triton optimizations.
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
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias_param', None)
            
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution using Triton kernels.
        """
        # Depthwise convolution using Triton
        x = triton_depthwise_conv2d(
            x, 
            self.depthwise_weight, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )
        
        # Pointwise convolution using Triton
        x = triton_pointwise_conv2d(
            x, 
            self.pointwise_weight, 
            self.bias_param
        )
        
        return x