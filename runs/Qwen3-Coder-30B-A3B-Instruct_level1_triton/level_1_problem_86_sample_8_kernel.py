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
    kernel_height,
    kernel_width,
    out_height,
    out_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    channel_block = tl.program_id(3)
    
    # Calculate output position
    out_h_start = out_h_idx * stride_h
    out_w_start = out_w_idx * stride_w
    
    # Calculate input boundaries with padding
    in_h_start = out_h_start - padding_h
    in_w_start = out_w_start - padding_w
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE, BLOCK_SIZE))
    
    # Process channels in blocks
    for c_offset in range(0, in_channels, CHANNELS_PER_BLOCK):
        if channel_block * CHANNELS_PER_BLOCK >= in_channels:
            break
            
        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        
        # Loop over kernel dimensions
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Calculate input positions
                in_h = in_h_start + kh * dilation_h
                in_w = in_w_start + kw * dilation_w
                
                # Load input data with boundary checking
                input_val = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
                if in_h >= 0 and in_h < height and in_w >= 0 and in_w < width:
                    input_val = tl.load(input_ptr + 
                                      batch_idx * (in_channels * height * width) +
                                      (channel_block * CHANNELS_PER_BLOCK) * (height * width) +
                                      in_h * width + in_w,
                                      mask=(in_h < height) & (in_w < width),
                                      other=0.0)
                
                # Load weight
                weight_val = tl.load(weight_ptr + 
                                   kh * kernel_width + kw,
                                   mask=(kh < kernel_height) & (kw < kernel_width),
                                   other=0.0)
                
                # Accumulate
                acc += input_val * weight_val
        
        # Store result
        output_offset = batch_idx * (in_channels * out_height * out_width) + \
                       (channel_block * CHANNELS_PER_BLOCK) * (out_height * out_width) + \
                       out_h_idx * out_width + out_w_idx
        tl.store(output_ptr + output_offset, acc, mask=True)

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
    # Get thread indices
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    w_idx = tl.program_id(2)
    out_ch_idx = tl.program_id(3)
    
    # Calculate output position
    output_offset = batch_idx * (out_channels * height * width) + \
                   out_ch_idx * (height * width) + \
                   h_idx * width + w_idx
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Compute dot product
    for i in range(in_channels):
        input_val = tl.load(input_ptr + 
                          batch_idx * (in_channels * height * width) +
                          i * (height * width) +
                          h_idx * width + w_idx)
        weight_val = tl.load(weight_ptr + 
                           out_ch_idx * in_channels + i)
        acc += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + output_offset, acc)

def triton_depthwise_conv2d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1):
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_height, kernel_width = weight.shape[-2], weight.shape[-1]
    
    # Calculate output dimensions
    out_height = (height + 2 * padding - (dilation * (kernel_height - 1) + 1)) // stride + 1
    out_width = (width + 2 * padding - (dilation * (kernel_width - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, out_height, out_width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    grid = (
        batch_size,
        out_height,
        out_width,
        math.ceil(in_channels / 32)
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
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride,
        stride,
        padding,
        padding,
        dilation,
        dilation,
        BLOCK_SIZE=16,
        CHANNELS_PER_BLOCK=32
    )
    
    return output

def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height, width, device=input_tensor.device, dtype=torch.float32)
    
    # Configure grid
    grid = (
        batch_size,
        height,
        width,
        out_channels
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
        BLOCK_SIZE=16
    )
    
    # Add bias if present
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
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
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
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
        # Use Triton kernel for depthwise convolution
        x = triton_depthwise_conv2d(
            x, 
            self.depthwise_weight, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )
        
        # Use Triton kernel for pointwise convolution
        x = triton_pointwise_conv2d(x, self.pointwise_weight, self.bias_param)
        
        return x