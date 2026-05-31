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
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    out_w = (width + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H + 2 * padding, BLOCK_SIZE_W + 2 * padding))
    
    # Calculate global indices
    global_h_start = h_idx * BLOCK_SIZE_H
    global_w_start = w_idx * BLOCK_SIZE_W
    
    # Load input tile with padding
    for i in range(BLOCK_SIZE_H + 2 * padding):
        for j in range(BLOCK_SIZE_W + 2 * padding):
            h = global_h_start + i - padding
            w = global_w_start + j - padding
            if 0 <= h < height and 0 <= w < width:
                val = tl.load(input_ptr + batch_idx * in_channels * height * width + 
                             channel_idx * height * width + h * width + w)
            else:
                val = 0.0
            shared_input[i * (BLOCK_SIZE_W + 2 * padding) + j] = val
    
    # Compute convolution for this block
    for i in range(BLOCK_SIZE_H):
        for j in range(BLOCK_SIZE_W):
            if i >= out_h or j >= out_w:
                continue
                
            h_out = h_idx * BLOCK_SIZE_H + i
            w_out = w_idx * BLOCK_SIZE_W + j
            
            acc = 0.0
            for k in range(kernel_size):
                for l in range(kernel_size):
                    # Apply dilation
                    kh = k * dilation
                    kw = l * dilation
                    
                    # Get input value from shared memory
                    ih = kh + h_out * stride
                    iw = kw + w_out * stride
                    
                    # Check bounds
                    if 0 <= ih < height + 2 * padding and 0 <= iw < width + 2 * padding:
                        input_val = shared_input[ih * (BLOCK_SIZE_W + 2 * padding) + iw]
                        weight_val = tl.load(weight_ptr + channel_idx * kernel_size * kernel_size + 
                                           k * kernel_size + l)
                        acc += input_val * weight_val
            
            # Store output
            tl.store(output_ptr + batch_idx * in_channels * out_h * out_w + 
                    channel_idx * out_h * out_w + h_out * out_w + w_out, acc)

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
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Calculate output dimensions
    out_h = height
    out_w = width
    
    # Shared memory for input tile
    shared_input = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE_H, BLOCK_SIZE_W))
    
    # Calculate global indices
    global_h_start = h_idx * BLOCK_SIZE_H
    global_w_start = w_idx * BLOCK_SIZE_W
    
    # Load input tile
    for i in range(BLOCK_SIZE_H):
        for j in range(BLOCK_SIZE_W):
            h = global_h_start + i
            w = global_w_start + j
            if h < out_h and w < out_w:
                for c_in in range(BLOCK_SIZE_C_IN):
                    if c_in < in_channels:
                        val = tl.load(input_ptr + batch_idx * in_channels * out_h * out_w + 
                                     c_in * out_h * out_w + h * out_w + w)
                        shared_input[i * BLOCK_SIZE_W + j] += val * tl.load(weight_ptr + 
                            c_out_idx * in_channels * out_h * out_w + 
                            c_in * out_h * out_w + h * out_w + w)
    
    # Store output
    for i in range(BLOCK_SIZE_H):
        for j in range(BLOCK_SIZE_W):
            h = global_h_start + i
            w = global_w_start + j
            if h < out_h and w < out_w:
                tl.store(output_ptr + batch_idx * out_channels * out_h * out_w + 
                        c_out_idx * out_h * out_w + h * out_w + w, 
                        shared_input[i * BLOCK_SIZE_W + j])

def triton_depthwise_conv2d(input_tensor, weight, stride=1, padding=0, dilation=1):
    """Custom Triton kernel for depthwise convolution"""
    batch_size, in_channels, height, width = input_tensor.shape
    kernel_size = weight.shape[-1]
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    out_w = (width + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, in_channels, out_h, out_w, dtype=torch.float32, device=input_tensor.device)
    
    # Configure block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16
    BLOCK_SIZE_K = 16
    
    # Grid configuration
    grid = (
        batch_size,
        in_channels,
        (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
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
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        BLOCK_SIZE_C,
        BLOCK_SIZE_K
    )
    
    return output

def triton_pointwise_conv2d(input_tensor, weight):
    """Custom Triton kernel for pointwise convolution"""
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height, width, dtype=torch.float32, device=input_tensor.device)
    
    # Configure block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C_IN = 16
    BLOCK_SIZE_C_OUT = 16
    
    # Grid configuration
    grid = (
        batch_size,
        out_channels,
        (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
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
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        BLOCK_SIZE_C_IN,
        BLOCK_SIZE_C_OUT
    )
    
    return output

class ModelNew(nn.Module):
    """
    Performs a depthwise-separable 2D convolution operation using custom Triton kernels.

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
        
        # Initialize weights
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.pointwise_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
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
        # Use Triton kernel for depthwise convolution
        x = triton_depthwise_conv2d(x, self.depthwise_weight, self.stride, self.padding, self.dilation)
        
        # Use Triton kernel for pointwise convolution
        x = triton_pointwise_conv2d(x, self.pointwise_weight)
        
        # Add bias if present
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
            
        return x