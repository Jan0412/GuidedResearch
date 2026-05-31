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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Block indices
    block_m_idx = tl.program_id(0)
    block_n_idx = tl.program_id(1)
    
    # Create offsets for rows and cols
    rows = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks for rows and cols
    mask_m = rows < M
    mask_n = cols < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Create offsets for A and B tiles
        a_offsets = rows[:, None] * stride_am + (k + tl.arange(0, BLOCK_K)[None, :]) * stride_ak
        b_offsets = (k + tl.arange(0, BLOCK_K)[:, None]) * stride_bk + cols[None, :] * stride_bn
        
        # Masks for the current K block
        mask_k = tl.arange(0, BLOCK_K) < (K - k)
        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]
        
        # Load tiles with masking
        a = tl.load(A_ptr + a_offsets, mask=mask_a, other=0.0)
        b = tl.load(B_ptr + b_offsets, mask=mask_b, other=0.0)
        
        # Accumulate dot product
        acc += tl.dot(a, b)
        
    # Store result
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn, acc, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions for matrix multiplication."
    
    C = torch.empty((M, N), dtype=A.dtype, device='cuda')
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_stages=2,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)