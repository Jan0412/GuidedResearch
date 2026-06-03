import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple

# Triton kernel for 2D convolution with specific parameters
@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor: [batch, in_channels, H, W]
    w_ptr,  # Weight tensor: [out_channels, in_channels, kH, kW]
    b_ptr,  # Bias tensor: [out_channels] (can be None)
    y_ptr,  # Output tensor: [batch, out_channels, out_H, out_W]
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    k_h: tl.constexpr,
    k_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Block size for batch dimension
    BLOCK_N: tl.constexpr,  # Block size for output channels
    BLOCK_K: tl.constexpr,  # Block size for accumulation (in_channels * k_h * k_w)
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    
    # Compute output spatial position (top-left corner of kernel)
    # We'll process multiple output positions per kernel invocation using tiling
    # For simplicity, we'll compute one output position per kernel instance for this example
    # But to make it efficient, we'll use tiling across output positions
    
    # Calculate output position
    out_h_idx = tl.program_id(2) // (out_w // BLOCK_N)
    out_w_idx = (tl.program_id(2) % (out_w // BLOCK_N)) * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Ensure we're within bounds
    out_w_mask = out_w_idx < out_w
    
    # Compute input spatial position (top-left corner of receptive field)
    in_h_start = out_h_idx * stride_h - pad_h
    in_w_start = out_w_idx * stride_w - pad_w
    
    # Create ranges for input positions and channels
    # We'll process channels and kernel positions in blocks for accumulation
    k_h_range = tl.arange(0, k_h)
    k_w_range = tl.arange(0, k_w)
    c_range = tl.arange(0, in_channels)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for c in range(in_channels):
        for kh in range(k_h):
            for kw in range(k_w):
                # Compute actual input position
                in_h_pos = in_h_start + kh
                in_w_pos = in_w_start + kw
                
                # Check if input position is valid
                in_h_valid = (in_h_pos >= 0) & (in_h_pos < in_h)
                in_w_valid = (in_w_pos >= 0) & (in_w_pos < in_w)
                valid_mask = in_h_valid & in_w_valid
                
                # Compute input pointer offset
                # Input layout: [batch, in_channels, in_h, in_w]
                # Offset = batch * (in_c * in_h * in_w) + c * (in_h * in_w) + in_h_pos * in_w + in_w_pos
                in_offset = (pid_batch * (in_channels * in_h * in_w) + 
                            c * (in_h * in_w) + 
                            in_h_pos * in_w + 
                            in_w_pos)
                
                # Load input value
                x_val = tl.load(x_ptr + in_offset, mask=valid_mask, other=0.0)
                
                # Compute weight pointer offset
                # Weight layout: [out_channels, in_channels, k_h, k_w]
                # Offset = out_ch * (in_c * k_h * k_w) + c * (k_h * k_w) + kh * k_w + kw
                w_offset = (pid_out_ch * (in_channels * k_h * k_w) + 
                           c * (k_h * k_w) + 
                           kh * k_w + 
                           kw)
                
                # Load weight value
                w_val = tl.load(w_ptr + w_offset)
                
                # Accumulate
                acc += x_val * w_val
    
    # Add bias if present
    if b_ptr is not None:
        bias = tl.load(b_ptr + pid_out_ch)
        acc += bias
    
    # Store result
    # Output layout: [batch, out_channels, out_h, out_w]
    out_offset = (pid_batch * (out_channels * out_h * out_w) + 
                 pid_out_ch * (out_h * out_w) + 
                 out_h_idx * out_w + 
                 out_w_idx)
    
    tl.store(y_ptr + out_offset, acc.to(tl.float32), mask=out_w_mask)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                 stride: Tuple[int, int] = (4, 4), padding: Tuple[int, int] = (2, 2)) -> torch.Tensor:
    """
    Custom Triton-based 2D convolution for the specific layer configuration.
    """
    # Get dimensions
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k_h, k_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    # Compute output dimensions
    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output tensor
    y = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Set kernel parameters
    BLOCK_M = 1  # Batch size block (we process one batch at a time for simplicity)
    BLOCK_N = 16  # Output channel block size
    BLOCK_K = 1  # Not used in this implementation but kept for consistency
    
    # Grid dimensions
    # [batch, out_channels, out_h * out_w // BLOCK_N]
    grid = (batch_size, out_channels, (out_h * out_w + BLOCK_N - 1) // BLOCK_N)
    
    # Launch kernel
    conv2d_kernel[grid](
        x, weight, bias, y,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k_h, k_w,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        # Use the original conv1 parameters but implement convolution via Triton
        return triton_conv2d(x, self.conv1.weight, self.conv1.bias,
                            stride=self.conv1.stride, padding=self.conv1.padding)