import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def pointwise_conv_kernel(
    x_ptr,          # Input tensor: (B, C_in, H, W)
    w_ptr,          # Weight tensor: (C_out, C_in, 1, 1)
    b_ptr,          # Bias tensor: (C_out,) or None
    out_ptr,        # Output tensor: (B, C_out, H, W)
    B, C_in, C_out, H, W,
    stride_x, stride_c_in, stride_h, stride_w,
    stride_w_out, stride_w_c_in, stride_w_h, stride_w_w,
    stride_b,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Program IDs for output tensor dimensions
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output spatial positions
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    
    # Create ranges for channels and spatial positions
    c_out_offsets = pid_c_out * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    c_in_offsets = tl.arange(0, BLOCK_SIZE_C_IN)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for bounds checking
    c_out_mask = c_out_offsets < C_out
    c_in_mask = c_in_offsets < C_in
    h_mask = h_offsets < H
    w_mask = w_offsets < W
    
    # Initialize accumulator for output
    acc = tl.zeros([BLOCK_SIZE_C_OUT, BLOCK_SIZE_H, BLOCK_SIZE_W], dtype=tl.float32)
    
    # Iterate over input channels
    for c_in_start in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_range = c_in_start + c_in_offsets
        c_in_m = c_in_range < C_in
        
        # Load input: shape (B, C_in, H, W)
        # For each c_in, we need to load [H, W] values for all c_out channels
        x_ptr_offset = pid_b * stride_x + c_in_range[:, None, None] * stride_c_in + \
                       h_offsets[None, :, None] * stride_h + w_offsets[None, None, :] * stride_w
        
        # We need to handle the case where BLOCK_SIZE_C_IN might be larger than remaining channels
        x_block = tl.load(x_ptr + x_ptr_offset, mask=c_in_m[:, None, None], other=0.0)
        
        # Load weights: shape (C_out, C_in, 1, 1)
        w_ptr_offset = c_out_offsets[:, None, None] * stride_w_out + \
                       c_in_range[None, :, None] * stride_w_c_in + \
                       0 * stride_w_h + 0 * stride_w_w
        
        w_block = tl.load(w_ptr + w_ptr_offset, mask=c_out_mask[:, None, None] & c_in_m[None, :, None], other=0.0)
        
        # Accumulate: matmul-like operation
        # x_block: [C_in, H, W], w_block: [C_out, C_in, 1]
        # We want: acc[c_out, h, w] += sum_cin(x[c_in, h, w] * w[c_out, c_in, 1, 1])
        acc += tl.sum(w_block[:, :, None, None] * x_block[None, :, :, :], axis=1)
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
        acc += bias[:, None, None]
    
    # Store output
    out_ptr_offset = pid_b * stride_x + c_out_offsets[:, None, None] * stride_c_in + \
                     h_offsets[None, :, None] * stride_h + w_offsets[None, None, :] * stride_w
    
    tl.store(out_ptr + out_ptr_offset, acc, mask=c_out_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


def triton_pointwise_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    Optimized pointwise 2D convolution using Triton.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    assert x.dim() == 4, "Input must be 4D tensor (B, C, H, W)"
    assert weight.dim() == 4, "Weight must be 4D tensor (C_out, C_in, 1, 1)"
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    B, C_in, H, W = x.shape
    C_out, _, _, _ = weight.shape
    
    # Create output tensor
    out = torch.empty(B, C_out, H, W, device=x.device, dtype=x.dtype)
    
    # Calculate strides
    stride_x = x.stride(0)
    stride_c_in = x.stride(1)
    stride_h = x.stride(2)
    stride_w = x.stride(3)
    
    stride_w_out = weight.stride(0)
    stride_w_c_in = weight.stride(1)
    stride_w_h = weight.stride(2)
    stride_w_w = weight.stride(3)
    
    stride_b = bias.stride(0) if bias is not None else 0
    
    # Grid dimensions: (batch, C_out blocks, H blocks, W blocks)
    BLOCK_SIZE_C_OUT = 8
    BLOCK_SIZE_C_IN = 8
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    
    grid = (
        B,
        (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,
        (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )
    
    # Launch kernel
    pointwise_conv_kernel[grid](
        x, weight, bias, out,
        B, C_in, C_out, H, W,
        stride_x, stride_c_in, stride_h, stride_w,
        stride_w_out, stride_w_c_in, stride_w_h, stride_w_w,
        stride_b,
        BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
        BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation using optimized Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        # Reuse the same weight and bias as the original conv1d
        self.weight = self.conv1d.weight
        self.bias = self.conv1d.bias if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using Triton kernel.
        """
        return triton_pointwise_conv(x, self.weight, self.bias)