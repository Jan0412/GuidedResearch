import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def pointwise_conv_kernel(
    x_ptr,  # Input tensor (B, C_in, H, W)
    w_ptr,  # Weight tensor (C_out, C_in)
    b_ptr,  # Bias tensor (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, H, W)
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    spatial_size,  # H * W
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)  # For output rows (C_out)
    pid_n = tl.program_id(1)  # For output columns (B*H*W)
    
    #_offsets for output matrix
    # pid_m corresponds to output channel, pid_n corresponds to spatial position
    # We treat this as matrix multiplication: (C_out, C_in) @ (C_in, B*H*W) = (C_out, B*H*W)
    
    # Offset for output
    out_offsets = pid_m * spatial_size + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create a range for output channels
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension (in_channels)
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load input: shape (C_in, B*H*W) but we need (C_in, BLOCK_SIZE_N)
        x_offsets = k * spatial_size + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        x_mask = (x_offsets < in_channels * spatial_size) & (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) < spatial_size * batch_size)
        
        # Actually we need to think in terms of (C_in, B*H*W) matrix
        # Let's restructure: x is stored as (B, C_in, H, W), so for position p = i*B*H*W + c*H*W + s
        # We need to compute: out[m, p] = sum_k w[m,k] * x[k, p]
        
        # For this kernel, let's compute for each output position in the batch spatial grid
        # pid_n covers multiple spatial positions across batches
        
        # Actually, let's use a simpler approach: we'll iterate over the batch and spatial positions
        
    # Alternative approach: use a different tiling strategy
    # Let pid_n represent a specific (batch, h, w) position
    # But that would be inefficient for large spatial sizes
    
    # Better approach: process one output channel at a time, compute across all batch*spatial positions
    
    # Let's change the grid: 
    # pid_m = output channel
    # pid_n = block of batch*spatial positions
    
    # For the kernel implementation, we'll compute:
    # out[batch, out_c, h, w] = sum_in_c w[out_c, in_c] * x[batch, in_c, h, w]
    
    # We can process multiple (batch, h, w) positions per block
    
    # Let's use the original grid idea but restructure the memory access
    # We'll treat it as: for each output channel m, compute over all spatial positions
    
    # Actually, let me redesign the kernel for better performance
    
    pass  # Placeholder, will implement below

@triton.jit
def pointwise_conv_kernel_v2(
    x_ptr,  # Input tensor (B, C_in, H, W)
    w_ptr,  # Weight tensor (C_out, C_in)
    b_ptr,  # Bias tensor (C_out,) or None
    out_ptr,  # Output tensor (B, C_out, H, W)
    batch_size,  # B
    in_channels,  # C_in
    out_channels,  # C_out
    height,  # H
    width,   # W
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial positions
    BLOCK_SIZE_K: tl.constexpr,  # Block size for input channels
):
    # Program IDs
    pid_m = tl.program_id(0)  # Output channel block
    pid_n = tl.program_id(1)  # Spatial position block
    
    # Calculate actual spatial position start
    spatial_pos_start = pid_n * BLOCK_SIZE_N
    
    # For output channel range
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < out_channels
    
    # For spatial positions
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    n_mask = spatial_pos_start + n_offsets < batch_size * height * width
    
    # Create mask for output
    out_mask = m_mask[:, None] & n_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over input channels
    for k in range(0, in_channels, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_channels
        
        # Load weights: shape (C_out, C_in), we need (BLOCK_SIZE_M, BLOCK_SIZE_K)
        w_offsets = m_offsets[:, None] * in_channels + k_offsets[None, :]
        w = tl.load(w_ptr + w_offsets, mask=k_mask[None, :] & m_mask[:, None], other=0.0)
        
        # Load inputs: shape (B, C_in, H, W), need (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # Convert spatial position to (batch, h, w)
        pos_offsets = spatial_pos_start + n_offsets
        batch_ids = pos_offsets // (height * width)
        h_offsets = (pos_offsets % (height * width)) // width
        w_offsets_spatial = pos_offsets % width
        
        # Calculate input offsets: [batch, channel, height, width]
        # Input layout: B * C_in * H * W, so offset = batch * (C_in*H*W) + channel * (H*W) + h * W + w
        x_offsets = (batch_ids[None, :] * in_channels * height * width + 
                    k_offsets[:, None] * height * width + 
                    h_offsets[None, :] * width + 
                    w_offsets_spatial[None, :])
        
        x = tl.load(x_ptr + x_offsets, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Accumulate: acc += w @ x^T (but here we're doing element-wise multiplication for each position)
        # Actually for pointwise conv: out[c_out, pos] += w[c_out, c_in] * x[c_in, pos]
        # So acc += w * x (broadcasted)
        acc += tl.dot(w, x, out_dtype=tl.float32)
    
    # Add bias if provided
    if b_ptr is not None:
        b = tl.load(b_ptr + m_offsets, mask=m_mask, other=0.0)
        acc += b[:, None]
    
    # Store output
    out_offsets = (m_offsets[:, None] * batch_size * height * width + 
                  spatial_pos_start + n_offsets[None, :])
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)

def pointwise_conv_triton(x, weight, bias=None):
    """Triton implementation of pointwise 2D convolution (1x1 conv)"""
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, _, _ = weight.shape
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Prepare output
    out = torch.empty(batch_size, out_channels, height, width, dtype=x.dtype, device=x.device)
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 32  # Output channels per block
    BLOCK_SIZE_N = 256  # Spatial positions per block
    BLOCK_SIZE_K = 16  # Input channels per block
    
    # Calculate grid dimensions
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (batch_size * height * width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    pointwise_conv_kernel_v2[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, height, width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized version of the pointwise 2D convolution using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # Register weights and bias as buffers/parameters to maintain compatibility
        # We'll use the same weight and bias as the original conv1d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = bias
        
        # Create the weight and bias parameters to be consistent with nn.Module
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized pointwise 2D convolution using Triton kernel.
        """
        # Ensure x is contiguous
        x = x.contiguous()
        
        # Call our custom Triton implementation
        return pointwise_conv_triton(x, self.weight, self.bias)