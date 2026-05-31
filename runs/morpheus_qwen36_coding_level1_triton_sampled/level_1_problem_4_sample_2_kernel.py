import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_offsets = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    mask_rows = row_offsets < M
    
    for start_k in range(0, K, BLOCK_K):
        mask_cols = (start_k + cols) < K
        mask = mask_rows[:, None] & mask_cols[None, :]
        a = tl.load(A_ptr + row_offsets[:, None] * K + start_k + cols, mask=mask, other=0.0)
        
        mask_b = (start_k + cols) < K
        b = tl.load(B_ptr + start_k + cols, mask=mask_b, other=0.0)
        b = tl.reshape(b, (BLOCK_K, 1))
        
        acc += tl.sum(tl.dot(a, b), axis=1)
        
    tl.store(C_ptr + row_offsets, acc, mask=mask_rows)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    assert B.shape == (K, 1), f"Expected B shape (K, 1), got {B.shape}"
    
    C = torch.empty((M, 1), dtype=torch.float32, device='cuda')
    
    BLOCK_M = 256
    BLOCK_K = 128
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M,)
    
    matmul_kernel[grid](A, B, C, M, K, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)