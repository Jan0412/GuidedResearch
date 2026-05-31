import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128

@triton.jit
def matmul_kernel(A_ptr, B_T_ptr, C_ptr, 
                  m, n, k, batch_size,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    i = tl.program_id(1)
    j = tl.program_id(2)
    
    c_offset = b * m * n + i * BLOCK_M * n + j * BLOCK_N
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)
    
    c_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_idx in range(0, k, BLOCK_K):
        a_offset = b * m * k + i * BLOCK_M * k + k_idx
        a_row_idx = tl.arange(0, BLOCK_M)
        a_col_idx = tl.arange(0, BLOCK_K)
        
        b_t_offset = b * n * k + j * BLOCK_N * k + k_idx
        b_t_row_idx = tl.arange(0, BLOCK_N)
        b_t_col_idx = tl.arange(0, BLOCK_K)
        
        a_tile = tl.load(A_ptr + a_offset + a_row_idx[:, None] * k + a_col_idx[None, :])
        b_t_tile = tl.load(B_T_ptr + b_t_offset + b_t_row_idx[:, None] * k + b_t_col_idx[None, :])
        
        c_tile += tl.dot(a_tile, b_t_tile.T)
    
    c_idx = c_offset + row_idx[:, None] * n + col_idx[None, :]
    tl.store(C_ptr + c_idx, c_tile)

def triton_bmm(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    B_T = B.transpose(1, 2).contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    C = torch.empty(batch_size, m, n, dtype=A.dtype, device=A.device)
    
    grid = (batch_size, m // BLOCK_M, n // BLOCK_N)
    
    matmul_kernel[grid](A, B_T, C, m, n, k, batch_size, BLOCK_M, BLOCK_N, BLOCK_K)
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_bmm(A, B)