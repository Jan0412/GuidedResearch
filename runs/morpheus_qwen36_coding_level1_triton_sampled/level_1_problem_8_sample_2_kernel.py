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
    block_idx_m = tl.program_id(0)
    block_idx_n = tl.program_id(1)
    
    offs_am = block_idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = block_idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_am < M
    mask_n = offs_bn < N
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        k_end = min(k_start + BLOCK_K, K)
        mask_k = offs_k < (k_end - k_start)
        
        a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, accumulator, mask=mask_c)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        K, 1,
        N, 1,
        N, 1,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)