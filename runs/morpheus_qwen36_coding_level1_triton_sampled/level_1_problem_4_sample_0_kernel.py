import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
        
    row_idx = pid
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    acc = 0.0
    
    for start_k in range(0, K, BLOCK_SIZE_K):
        col_offsets = start_k + k_offsets
        mask = col_offsets < K
        
        a = tl.load(A_ptr + row_idx * K + col_offsets, mask=mask, other=0.0)
        b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
        
        acc += tl.sum(a * b)
        
    tl.store(C_ptr + row_idx, acc)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    BLOCK_SIZE_K = 1024
    
    grid = (M,)
    matmul_kernel[grid](A, B, C, M, K, BLOCK_SIZE_K=BLOCK_SIZE_K)
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)