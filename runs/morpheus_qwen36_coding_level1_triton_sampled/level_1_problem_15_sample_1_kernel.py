import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triangular_matmul_kernel(
    A_ptr, B_T_ptr, C_ptr, M, BLOCK_SIZE: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    row_off = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    col_off = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    
    row_mask = row_off < M
    col_mask = col_off < M
    valid_mask = row_mask & col_mask & (row_off >= col_off)
    
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    for k_start in range(0, M, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < M
        
        A_vals = tl.load(A_ptr + row_off * M + k_off, mask=k_mask[None, :], other=0.0)
        B_vals = tl.load(B_T_ptr + col_off * M + k_off, mask=k_mask[None, :], other=0.0)
        
        acc += tl.sum(A_vals * B_vals, axis=1)[:, None]
        
    tl.store(C_ptr + row_off * M + col_off, acc, mask=valid_mask)


def triton_triangular_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    M = A.shape[0]
    B_T = B.t().contiguous()
    C = torch.zeros_like(A)
    BLOCK_SIZE = 64
    BLOCK_K = 64
    N_BLOCKS = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (N_BLOCKS, N_BLOCKS)
    triangular_matmul_kernel[grid](A, B_T, C, M, BLOCK_SIZE, BLOCK_K)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_triangular_matmul(A, B)