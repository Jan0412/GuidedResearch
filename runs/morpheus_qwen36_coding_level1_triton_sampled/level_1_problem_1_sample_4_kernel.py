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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    mask_am = offsets_am < M
    mask_bn = offsets_bn < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a_ptrs = A + offsets_am[:, None] * stride_am + (k + offsets_k[None, :]) * stride_ak
        b_ptrs = B + (k + offsets_k[:, None]) * stride_bk + offsets_bn[None, :] * stride_bn
        
        a_mask = mask_am[:, None] & ((k + offsets_k[None, :]) < K)
        b_mask = ((k + offsets_k[:, None]) < K) & mask_bn[None, :]
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
        
    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_c = mask_am[:, None] & mask_bn[None, :]
    
    c_ptrs = C + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_c)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert A.shape[1] == B.shape[0]
    
    A = A.contiguous()
    B = B.contiguous()
    
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_SIZE = 128
    
    grid = ((M + BLOCK_SIZE - 1) // BLOCK_SIZE, 
            (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
            
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE,
        BLOCK_SIZE_N=BLOCK_SIZE,
        BLOCK_SIZE_K=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)