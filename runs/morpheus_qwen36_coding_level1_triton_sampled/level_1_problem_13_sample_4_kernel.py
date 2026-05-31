import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    m_mask = offs_m < N
    n_mask = offs_n < N
    
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_SIZE_K):
        a_ptrs = A_ptr + offs_m[:, None] * N + (k + offs_k)[None, :]
        a_mask = m_mask[:, None] & (k + offs_k)[None, :] < N
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = B_ptr + (k + offs_k)[:, None] * N + offs_n[None, :]
        b_mask = (k + offs_k)[:, None] < N & n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        c += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)
        
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty_like(A)
    
    N = A.shape[0]
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    grid = lambda meta: (
        (N + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
        1
    )
    
    matmul_kernel[grid](A, B, C, N, 
                          BLOCK_SIZE_M=BLOCK_SIZE_M, 
                          BLOCK_SIZE_N=BLOCK_SIZE_N, 
                          BLOCK_SIZE_K=BLOCK_SIZE_K)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)