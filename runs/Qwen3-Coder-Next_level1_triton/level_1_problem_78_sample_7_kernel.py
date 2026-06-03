import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def transposed_conv2d_kernel(
    input_ptr,          # Input tensor (N, C_in, H_in, W_in)
    weight_ptr,         # Weight tensor (C_in, C_out, K_h, K_w)
    bias_ptr,           # Bias tensor (C_out,)
    output_ptr,         # Output tensor (N, C_out, H_out, W_out)
    N, C_in, H_in, W_in,
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    out_h, out_w,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_COUT: tl.constexpr,
    BLOCK_SIZE_CIN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    # Get batch and output channel indices
    batch_idx = tl.program_id(0)
    out_c_idx = tl.program_id(1) * BLOCK_SIZE_COUT + tl.arange(0, BLOCK_SIZE_COUT)
    
    # Output spatial coordinates
    out_h_idx = tl.program_id(2) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for output channels and spatial positions
    c_out_mask = out_c_idx < C_out
    h_mask = out_h_idx < out_h
    
    # Initialize accumulator for output
    output_sum = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_COUT), dtype=tl.float32)
    
    # Iterate over input channels and kernel dimensions
    for in_c in range(C_in):
        for kh in range(K_h):
            for kw in range(K_w):
                # Calculate corresponding input position
                in_h = out_h_idx * stride_h + kh - pad_h
                in_w = tl.arange(0, 1) * stride_w + kw - pad_w  # Will be broadcast
                
                # Check if input position is valid
                valid_h = (in_h >= 0) & (in_h < H_in)
                valid_w = (in_w >= 0) & (in_w < W_in)
                valid = valid_h & valid_w
                
                # Load input value (broadcast across width dimension)
                if in_w[0] >= 0 and in_w[0] < W_in:
                    in_ptr_offset = batch_idx * (C_in * H_in * W_in) + \
                                   in_c * (H_in * W_in) + \
                                   in_h * W_in + \
                                   in_w[0]
                    input_val = tl.load(input_ptr + in_ptr_offset, mask=valid[:, 0], other=0.0)
                else:
                    input_val = 0.0
                
                # Load weight value
                weight_ptr_offset = in_c * (C_out * K_h * K_w) + \
                                   out_c_idx[:, None] * (K_h * K_w) + \
                                   kh * K_w + \
                                   kw
                weight_val = tl.load(weight_ptr + weight_ptr_offset, mask=c_out_mask[:, None], other=0.0)
                
                # Accumulate
                output_sum += input_val[:, None] * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_c_idx, mask=c_out_mask, other=0.0)
        output_sum += bias[None, :]
    
    # Store output
    output_ptr_offset = batch_idx * (C_out * out_h * out_w) + \
                       out_c_idx[None, :] * (out_h * out_w) + \
                       out_h_idx[:, None] * out_w
    
    # Store with proper masks
    mask = h_mask[:, None] & c_out_mask[None, :]
    tl.store(output_ptr + output_ptr_offset, output_sum, mask=mask)


class TritonTransposedConv2d(nn.Module):
    """Triton implementation of 2D transposed convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        
        # Initialize with kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape
        assert in_channels == self.in_channels, f"Expected {self.in_channels} channels, got {in_channels}"
        
        # Calculate output dimensions
        out_h = (in_h - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0]
        out_w = (in_w - 1) * self.stride[1] - 2 * self.padding[1] + self.kernel_size[1]
        
        # Allocate output tensor
        output = torch.empty(batch_size, self.out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
        
        # Launch kernel with tuned block sizes
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_COUT = 16
        BLOCK_SIZE_CIN = 8
        BLOCK_SIZE_KH = 2
        BLOCK_SIZE_KW = 4
        
        grid = (
            batch_size,  # batch dimension
            (self.out_channels + BLOCK_SIZE_COUT - 1) // BLOCK_SIZE_COUT,  # output channels
            (out_h + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N  # height dimension
        )
        
        # Ensure tensors are contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None
        
        transposed_conv2d_kernel[grid](
            x, weight, bias, output,
            batch_size, self.in_channels, in_h, in_w,
            self.out_channels, self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            out_h, out_w,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_COUT=BLOCK_SIZE_COUT,
            BLOCK_SIZE_CIN=BLOCK_SIZE_CIN,
            BLOCK_SIZE_KH=BLOCK_SIZE_KH,
            BLOCK_SIZE_KW=BLOCK_SIZE_KW,
        )
        
        return output


import math


class ModelNew(nn.Module):
    """
    Optimized version of the transposed convolution model using custom Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = TritonTransposedConv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d(x)