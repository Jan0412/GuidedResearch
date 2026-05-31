import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    
    offs_am = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    mask_m = offs_am < N
    mask_n = offs_bn < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_K):
        offs_ak = k + offs_k
        mask_k = offs_ak < N
        
        a_block = tl.load(A_ptr + offs_am[:, None] * N + offs_ak[None, :], 
                          mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b_block = tl.load(B_ptr + offs_ak[:, None] * N + offs_bn[None, :], 
                          mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        
        acc = tl.dot(a_block, b_block, acc, allow_tf32=False)
        
    tl.store(C_ptr + offs_am[:, None] * N + offs_bn[None, :], 
             acc, mask=(mask_m[:, None] & mask_n[None, :]))


def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    N = A.shape[0]
    C = torch.empty_like(A)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    def grid(meta):
        return ((N + meta["BLOCK_M"] - 1) // meta["BLOCK_M"], 
                (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"], 1)
    
    matmul_kernel[grid](A, B, C, N, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4, num_stages=2)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)