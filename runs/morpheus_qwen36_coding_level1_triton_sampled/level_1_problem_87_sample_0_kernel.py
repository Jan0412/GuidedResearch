import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_bias_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_x, stride_w, stride_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Program ID and block coordinates
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    block_m = pid // num_pid_n
    block_n = pid % num_pid_n
    
    row_off = block_m * BLOCK_M
    col_off = block_n * BLOCK_N
    
    # Indices for current block
    row_indices = row_off + tl.arange(0, BLOCK_M)
    col_indices = col_off + tl.arange(0, BLOCK_N)
    k_indices = tl.arange(0, BLOCK_K)
    
    # Masks for bounds checking
    mask_row = row_indices < M
    mask_col = col_indices < N
    mask_k = k_indices < K
    
    # Load input blocks with masking
    x_block = tl.load(
        x_ptr + row_indices[:, None] * stride_x + k_indices[None, :],
        mask=mask_row[:, None] & mask_k[None, :],
        other=0.0
    )
    w_block = tl.load(
        w_ptr + k_indices[:, None] * stride_w + col_indices[None, :],
        mask=mask_col[None, :] & mask_k[:, None],
        other=0.0
    )
    
    # Matrix multiplication
    acc = tl.dot(x_block, w_block)
    
    # Add bias if present
    if HAS_BIAS:
        bias_block = tl.load(bias_ptr + col_indices, mask=mask_col, other=0.0)
        acc += bias_block[None, :]
        
    # Store result
    out_indices = row_indices[:, None] * stride_out + col_indices[None, :]
    tl.store(
        out_ptr + out_indices,
        acc,
        mask=mask_row[:, None] & mask_col[None, :]
    )


def triton_conv2d_pointwise(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Custom Triton implementation of pointwise 2D convolution (1x1 Conv).
    Equivalent to batched matrix multiplication: Y = X @ W^T + Bias
    """
    # Ensure FP32 precision
    x = x.float()
    weight = weight.float()
    
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Flatten spatial dimensions: (B, C_in, H, W) -> (B*H*W, C_in)
    x_flat = x.reshape(B * H * W, C_in)
    
    # Transpose weight to match matmul layout: (C_out, C_in) -> (C_in, C_out)
    w_t = weight.t()
    
    # Prepare output tensor
    out_flat = torch.empty(B * H * W, C_out, dtype=torch.float32, device=x.device)
    
    M = B * H * W
    N = C_out
    K = C_in
    
    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # Grid calculation
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)
    
    # Strides
    stride_x = K
    stride_w = N
    stride_out = N
    
    # Launch kernel
    matmul_bias_kernel[grid](
        x_flat, w_t, bias, out_flat,
        M, N, K,
        stride_x, stride_w, stride_out,
        BLOCK_M, BLOCK_N, BLOCK_K,
        bias is not None
    )
    
    # Reshape back to (B, C_out, H, W)
    return out_flat.reshape(B, C_out, H, W)


class ModelNew(nn.Module):
    """
    Optimized pointwise 2D convolution using custom Triton kernels.
    Replaces nn.Conv2d with a fused matmul+bias kernel for speedup.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.has_bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d_pointwise(x, self.weight, self.bias if self.has_bias else None)