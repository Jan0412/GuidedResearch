import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID to block coordinates
    row_idx = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_idx = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Boundary masks
    mask_m = row_idx < M
    mask_n = col_idx < N
    
    # Accumulator for FP32 results
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_idx = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_idx < K
        
        # Compute pointers for A block (K, M) -> index = k * M + m
        a_ptrs = A_ptr + k_idx[:, None] * stride_am + row_idx[None, :] * stride_ak
        # Compute pointers for B block (K, N) -> index = k * N + n
        b_ptrs = B_ptr + k_idx[:, None] * stride_bk + col_idx[None, :] * stride_bn
        
        # Combine masks for loading
        mask_a = mask_k[:, None] & mask_m[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]
        
        # Load blocks with zero-padding for out-of-bound elements
        a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Matrix multiply-accumulate using hardware-optimized tl.dot
        acc += tl.dot(a_block, b_block)
        
    # Store result block
    c_ptrs = C_ptr + row_idx[:, None] * stride_cm + col_idx[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton matrix multiplication kernel.
    Handles contiguity, grid calculation, and kernel launch.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Extract dimensions based on input shapes (K, M) and (K, N)
    M = A.shape[1]
    K = A.shape[0]
    N = B.shape[1]
    
    # Allocate output tensor of shape (M, N)
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    # Define strides for row-major layout
    stride_am = M
    stride_ak = 1
    stride_bk = N
    stride_bn = 1
    stride_cm = N
    stride_cn = 1
    
    # Tunable block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    # Calculate grid dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A.T @ B using a custom Triton kernel.
        A: (K, M), B: (K, N) -> C: (M, N)
        """
        return triton_matmul(A, B)