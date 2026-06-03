import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Input tensor: (batch, channels, height, width)
    w_ptr,  # Weight tensor: (channels, kernel_h, kernel_w)
    b_ptr,  # Bias tensor: (channels,) or None
    out_ptr,  # Output tensor: (batch, channels, out_h, out_w)
    batch_size, in_channels, height, width, out_h, out_w,
    kernel_size, stride, padding,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
):
    # Program IDs for output spatial dimensions
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index
    pid_h = tl.program_id(2)  # output height block index
    pid_w = tl.program_id(3)  # output width block index
    
    # Calculate output position
    out_h_start = pid_h * BLOCK_SIZE_H
    out_w_start = pid_w * BLOCK_SIZE_W
    
    # Create ranges for output positions
    out_h_range = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_w_range = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    out_h_mask = out_h_range < out_h
    out_w_mask = out_w_range < out_w
    
    # Calculate input position offsets
    in_h_start = out_h_start * stride - padding
    in_w_start = out_w_start * stride - padding
    
    # Create kernel offsets
    kernel_h_range = tl.arange(0, KERNEL_H)
    kernel_w_range = tl.arange(0, KERNEL_W)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Iterate over kernel positions
    for kh in range(KERNEL_H):
        in_h = in_h_start + kh
        in_h_valid = (in_h >= 0) & (in_h < height)
        
        for kw in range(KERNEL_W):
            in_w = in_w_start + kw
            in_w_valid = (in_w >= 0) & (in_w < width)
            
            # Load input values with masking
            in_h_idx = in_h if in_h_valid else 0
            in_w_idx = in_w if in_w_valid else 0
            
            # Load input: shape (BLOCK_SIZE_H, BLOCK_SIZE_W)
            x_offset = (pid_b * in_channels * height * width + 
                       pid_c * height * width + 
                       in_h_idx * width + 
                       in_w_idx)
            x_vals = tl.load(
                x_ptr + x_offset,
                mask=(out_h_mask & out_w_mask) & in_h_valid & in_w_valid,
                other=0.0
            )
            
            # Load weight
            w_offset = pid_c * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw
            w_val = tl.load(w_ptr + w_offset)
            
            # Accumulate
            accumulator += x_vals * w_val
    
    # Apply bias if available
    if b_ptr is not None:
        b_offset = pid_c
        bias = tl.load(b_ptr + b_offset)
        accumulator += bias
    
    # Store result
    out_offset = (pid_b * in_channels * out_h * out_w + 
                 pid_c * out_h * out_w + 
                 out_h_range[:, None] * out_w + 
                 out_w_range[None, :])
    tl.store(
        out_ptr + out_offset,
        accumulator,
        mask=out_h_mask & out_w_mask
    )


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Performs depthwise 2D convolution using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch, channels, height, width)
        weight: Weight tensor of shape (channels, kernel_h, kernel_w)
        bias: Optional bias tensor of shape (channels,)
        stride: Stride of convolution
        padding: Padding applied to input
    
    Returns:
        Output tensor of shape (batch, channels, out_h, out_w)
    """
    batch_size, in_channels, height, width = x.shape
    _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_h = (height + 2 * padding - kernel_h) // stride + 1
    out_w = (width + 2 * padding - kernel_w) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    
    # Configure grid
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    grid = (
        batch_size,  # batch
        in_channels,  # channels
        (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,  # out_h blocks
        (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,  # out_w blocks
    )
    
    # Launch kernel
    depthwise_conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, height, width, out_h, out_w,
        kernel_h, stride, padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        KERNEL_H=kernel_h, KERNEL_W=kernel_w,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using Triton kernel.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size))
        
        # Initialize bias if needed
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)


import math