import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = rows < M
    mask_n = cols < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        cols_k = k + tl.arange(0, BLOCK_K)
        mask_k = cols_k < K
        
        a_ptrs = A + rows[:, None] * stride_am + cols_k[None, :] * stride_ak
        b_ptrs = B + cols_k[:, None] * stride_bk + cols[None, :] * stride_bn
        
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        
        acc += tl.dot(a, b)
        
    out_ptrs = C + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(out_ptrs, acc, mask=mask)

def triton_matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        1,
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []