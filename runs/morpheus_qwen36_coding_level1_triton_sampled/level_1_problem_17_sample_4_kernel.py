import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr, B_T_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bTm, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    mask_m = m_offsets < M
    mask_n = n_offsets < N
    mask_k = k_offsets < K
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a_offsets = m_offsets[:, None] * stride_am + (k + k_offsets)[None, :] * stride_ak
        a_mask = mask_m[:, None] & (k + k_offsets)[None, :] < K
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        bT_offsets = (k + k_offsets)[:, None] * stride_bTm + n_offsets[None, :] * stride_bTn
        bT_mask = (k + k_offsets)[:, None] < K & mask_n[None, :]
        bT = tl.load(B_T_ptr + bT_offsets, mask=bT_mask, other=0.0)
        
        acc += tl.matmul(a, bT)
        
    c_offsets = m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


def triton_gemm(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B_T = B.T.contiguous()
    
    M, K = A.shape
    N, _ = B.shape
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    stride_am = K
    stride_ak = 1
    stride_bTm = N
    stride_bTn = 1
    stride_cm = N
    stride_cn = 1
    
    gemm_kernel[grid](
        A, B_T, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bTm, stride_bTn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, A, B):
        return triton_gemm(A, B)