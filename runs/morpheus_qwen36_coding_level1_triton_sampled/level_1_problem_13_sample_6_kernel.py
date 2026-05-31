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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_row = row_offsets < N
    mask_col = col_offsets < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_k = (N + BLOCK_K - 1) // BLOCK_K
    for k in range(num_k):
        k_offsets = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < N
        
        a_offsets = row_offsets[:, None] * N + k_offsets[None, :]
        a_mask = mask_row[:, None] & mask_k[None, :]
        A_block = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        b_offsets = k_offsets[:, None] * N + col_offsets[None, :]
        b_mask = mask_k[:, None] & mask_col[None, :]
        B_block = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        acc += tl.dot(A_block, B_block)
        
    c_offsets = row_offsets[:, None] * N + col_offsets[None, :]
    c_mask = mask_row[:, None] & mask_col[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    N = A.shape[0]
    out = torch.empty_like(A)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid = (N // BLOCK_M, N // BLOCK_N)
    
    matmul_kernel[grid](A, B, out, N, BLOCK_M, BLOCK_N, BLOCK_K)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)