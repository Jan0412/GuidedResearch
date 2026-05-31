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
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
    OUTPUT_ELEMENTS_PER_BLOCK: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    
    # Calculate output dimensions
    output_x = tl.program_id(3)
    
    # Shared memory for input tile
    shared_input = tl.shared_tensor(tl.float32, (KERNEL_HEIGHT, KERNEL_WIDTH))
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over groups
    group_size = in_channels // groups
    group_start = out_channel_idx * group_size // out_channels * groups
    
    # Loop over input channels (grouped)
    for c in range(0, group_size, CHANNELS_PER_BLOCK):
        # Load weight
        w_offset = out_channel_idx * group_size + (c % group_size) * kernel_height * kernel_width
        w = tl.load(weight_ptr + w_offset, mask=(c % group_size) < group_size)
        
        # Load input tile (with padding)
        input_y_start = output_y * stride_h - padding_h
        input_x_start = output_x * stride_w - padding_w
        
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                input_y = input_y_start + kh * dilation_h
                input_x = input_x_start + kw * dilation_w
                
                # Check bounds
                if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                    input_val = tl.load(input_ptr + 
                                      batch_idx * in_channels * input_height * input_width +
                                      (group_start + c % group_size) * input_height * input_width +
                                      input_y * input_width + input_x)
                    acc += input_val * w
                    
        # Update weight pointer for next iteration
        w_offset += CHANNELS_PER_BLOCK * kernel_height * kernel_width
        
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    # Store result
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_channel_idx * output_height * output_width + \
                   output_y * output_width + output_x
    tl.store(output_ptr + output_offset, acc)

def triton_conv2d(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Triton implementation of 2D convolution with fused operations.
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare grid
    grid = (
        batch_size,
        out_channels,
        output_height,
        output_width
    )
    
    # Define block sizes
    BLOCK_SIZE = 128
    CHANNELS_PER_BLOCK = 8
    OUTPUT_ELEMENTS_PER_BLOCK = 32
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK,
        OUTPUT_ELEMENTS_PER_BLOCK=OUTPUT_ELEMENTS_PER_BLOCK
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Use Triton kernel for convolution
        return triton_conv2d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

# Simplified version for better performance
@triton.jit
def conv2d_simple_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
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
    BLOCK_SIZE: tl.constexpr
):
    # Thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    output_x = tl.program_id(3)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate group info
    group_size = in_channels // groups
    group_start = (out_channel_idx // (out_channels // groups)) * group_size
    
    # Loop over kernel and input channels
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            for c in range(group_size):
                # Calculate input coordinates
                input_y = output_y * stride_h - padding_h + kh * dilation_h
                input_x = output_x * stride_w - padding_w + kw * dilation_w
                
                # Bounds checking
                if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                      batch_idx * in_channels * input_height * input_width +
                                      (group_start + c) * input_height * input_width +
                                      input_y * input_width + input_x)
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + 
                                       out_channel_idx * group_size * kernel_height * kernel_width +
                                       c * kernel_height * kernel_width +
                                       kh * kernel_width + kw)
                    
                    acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    # Store result
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_channel_idx * output_height * output_width + \
                   output_y * output_width + output_x
    tl.store(output_ptr + output_offset, acc)

# Final optimized version using more efficient memory access patterns
@triton.jit
def conv2d_optimized_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
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
    BLOCK_SIZE: tl.constexpr
):
    # Thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    output_y = tl.program_id(2)
    output_x = tl.program_id(3)
    
    # Shared memory for weight loading
    shared_weight = tl.shared_tensor(tl.float32, (kernel_height, kernel_width))
    
    # Accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Group calculation
    group_size = in_channels // groups
    group_idx = out_channel_idx // (out_channels // groups)
    channel_start = group_idx * group_size
    
    # Loop over kernel and input channels
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            for c in range(group_size):
                # Calculate input position
                input_y = output_y * stride_h - padding_h + kh * dilation_h
                input_x = output_x * stride_w - padding_w + kw * dilation_w
                
                # Bounds check
                if input_y >= 0 and input_y < input_height and input_x >= 0 and input_x < input_width:
                    # Load input
                    input_val = tl.load(input_ptr + 
                                      batch_idx * in_channels * input_height * input_width +
                                      (channel_start + c) * input_height * input_width +
                                      input_y * input_width + input_x)
                    
                    # Load weight
                    weight_val = tl.load(weight_ptr + 
                                       out_channel_idx * group_size * kernel_height * kernel_width +
                                       c * kernel_height * kernel_width +
                                       kh * kernel_width + kw)
                    
                    acc += input_val * weight_val
    
    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    # Store output
    output_offset = batch_idx * out_channels * output_height * output_width + \
                   out_channel_idx * output_height * output_width + \
                   output_y * output_width + output_x
    tl.store(output_ptr + output_offset, acc)

def triton_conv2d_optimized(input_tensor, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Optimized Triton implementation of 2D convolution.
    """
    batch_size, in_channels, input_height, input_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * padding[0] - (dilation[0] * (kernel_height - 1) + 1)) // stride[0] + 1
    output_width = (input_width + 2 * padding[1] - (dilation[1] * (kernel_width - 1) + 1)) // stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
    
    # Prepare grid
    grid = (
        batch_size,
        out_channels,
        output_height,
        output_width
    )
    
    # Launch kernel
    conv2d_optimized_kernel[grid](
        input_tensor,
        weight,
        output,
        bias,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        BLOCK_SIZE=128
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.
    Optimized with custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias = bias
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Use optimized Triton kernel for convolution
        return triton_conv2d_optimized(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )