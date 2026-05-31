import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def pointwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    bias_enabled: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    
    # Calculate global indices
    h_start = tl.program_id(2) * stride_h
    w_start = tl.program_id(3) * stride_w
    
    # Shared memory for input tile
    input_tile = tl.shared_tile(input_ptr, [BLOCK_SIZE, BLOCK_SIZE], [1, 1])
    
    # Loop over input channels
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Load weight for this output channel
    weight_row = tl.load(weight_ptr + out_channel_id * in_channels + tl.arange(0, in_channels))
    
    # Process each element in the output tile
    for h_offset in range(0, stride_h):
        for w_offset in range(0, stride_w):
            h = h_start + h_offset
            w = w_start + w_offset
            
            # Check bounds
            if h < height and w < width:
                # Load input value
                input_val = tl.load(input_ptr + batch_id * in_channels * height * width + 
                                  tl.arange(0, in_channels) * height * width + 
                                  h * width + w)
                
                # Multiply with weights and accumulate
                acc += input_val * weight_row
    
    # Apply bias if enabled
    if bias_enabled:
        bias_val = tl.load(bias_ptr + out_channel_id)
        acc += bias_val
    
    # Store result
    output_idx = batch_id * out_channels * height * width + out_channel_id * height * width + \
                 h_start * width + w_start
    tl.store(output_ptr + output_idx, acc)

# Simplified approach using direct matrix multiplication for 1x1 convolutions
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    # Compute block IDs
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Tile ID within group
    group_id = pid // GROUP_M
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = pid // group_size_m
    
    # Pointers for A and B
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulation buffer
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiply
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator)

# Optimized version for 1x1 convolutions using fused matmul
def triton_pointwise_conv2d(input_tensor, weight, bias=None):
    """
    Optimized pointwise 2D convolution using Triton kernels
    """
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Reshape input for matrix multiplication
    # Input: (batch_size, in_channels, height, width) -> (batch_size * height * width, in_channels)
    input_reshaped = input_tensor.permute(0, 2, 3, 1).contiguous().view(-1, in_channels)
    
    # Weight: (out_channels, in_channels) -> (in_channels, out_channels) for matmul
    weight_t = weight.t()
    
    # Output: (batch_size * height * width, out_channels)
    output_reshaped = torch.empty(batch_size * height * width, out_channels, device=input_tensor.device, dtype=torch.float32)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 128
    M, K = input_reshaped.shape
    K, N = weight_t.shape
    
    # Grid dimensions
    grid_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_m * grid_n,)
    
    # Launch kernel
    matmul_kernel[grid](
        input_reshaped,
        weight_t,
        output_reshaped,
        M, N, K,
        input_reshaped.stride(0), input_reshaped.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        output_reshaped.stride(0), output_reshaped.stride(1),
        BLOCK_M=BLOCK_SIZE,
        BLOCK_N=BLOCK_SIZE,
        BLOCK_K=BLOCK_SIZE,
        GROUP_M=8
    )
    
    # Add bias if present
    if bias is not None:
        output_reshaped += bias
    
    # Reshape back to original dimensions
    output = output_reshaped.view(batch_size, height, width, out_channels).permute(0, 3, 1, 2)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized version of the Model class using Triton kernels for pointwise 2D convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bias_enabled = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Use the Triton-optimized implementation
        return triton_pointwise_conv2d(x, self.conv1d.weight, self.conv1d.bias if self.bias_enabled else None)