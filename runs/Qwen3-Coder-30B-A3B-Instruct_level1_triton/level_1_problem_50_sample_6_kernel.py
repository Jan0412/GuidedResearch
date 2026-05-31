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
    input_height,
    input_width,
    output_height,
    output_width,
    in_channels,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Tile indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    k_start = pid_k * BLOCK_SIZE_K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, in_channels * kernel_h * kernel_w, BLOCK_SIZE_K):
        # Load input tile
        input_tile = tl.load(input_ptr + 
            (m_start * stride_h + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_h + tl.arange(0, kernel_h)[None, :]) * input_width +
            (k_start * stride_w + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_w + tl.arange(0, kernel_w)[None, :]))
        
        # Load weight tile
        weight_tile = tl.load(weight_ptr + 
            (n_start + tl.arange(0, BLOCK_SIZE_N)[:, None]) * in_channels * kernel_h * kernel_w +
            (k_start + tl.arange(0, BLOCK_SIZE_K)[None, :]))
        
        # Accumulate
        acc += tl.dot(input_tile, weight_tile)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + n_start + tl.arange(0, BLOCK_SIZE_N))
        acc += bias[None, :]
    
    # Store output
    tl.store(output_ptr + 
        (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None]) * output_width + 
        (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]), 
        acc)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        
    def forward(self, x):
        # Use custom Triton implementation for convolution
        return self.triton_conv2d(x, self.conv1.weight, self.conv1.bias)
    
    def triton_conv2d(self, input_tensor, weight, bias=None):
        # Get dimensions
        batch_size, in_channels, input_height, input_width = input_tensor.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * 2 - kernel_h) // 4 + 1
        output_width = (input_width + 2 * 2 - kernel_h) // 4 + 1
        
        # Allocate output tensor
        output = torch.empty(batch_size, out_channels, output_height, output_width, device=input_tensor.device, dtype=torch.float32)
        
        # Define block sizes
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 32
        
        # Grid dimensions
        grid_m = (output_height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        grid_k = (in_channels * kernel_h * kernel_w + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        
        grid = (grid_m, grid_n, grid_k)
        
        # Launch kernel
        conv2d_kernel[grid](
            input_tensor,
            weight,
            output,
            bias,
            input_height,
            input_width,
            output_height,
            output_width,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            4,  # stride_h
            4,  # stride_w
            2,  # padding_h
            2,  # padding_w
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K
        )
        
        return output