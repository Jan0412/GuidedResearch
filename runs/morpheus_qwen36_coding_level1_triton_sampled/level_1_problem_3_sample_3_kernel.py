import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, m, k, n, batch_stride_a, batch_stride_b, batch_stride_c, BLOCK_M, BLOCK_K, BLOCK_N):
    batch_idx = tl.program_id(0)
    base_a = batch_idx * batch_stride_a
    base_b = batch_idx * batch_stride_b
    base_c = batch_idx * batch_stride_c
    
    off_i = tl.arange(0, BLOCK_M)
    off_j = tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for p in range(0, k, BLOCK_K):
        a_ptrs = A_ptr + base_a + off_i[:, None] * k + p + off_k[None, :]
        b_ptrs = B_ptr + base_b + (p + off_k[:, None]) * n + off_j[None, :]
        
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        
        acc = acc + tl.dot(a_block, b_block)
        
    c_ptrs = C_ptr + base_c + off_i[:, None] * n + off_j[None, :]
    tl.store(c_ptrs, acc)

def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device='cuda')
    
    batch_stride_a = m * k
    batch_stride_b = k * n
    batch_stride_c = m * n
    
    BLOCK_M = 128
    BLOCK_K = 128
    BLOCK_N = 128
    
    grid = (batch_size,)
    
    matmul_kernel[grid](
        A, B, C, m, k, n, batch_stride_a, batch_stride_b, batch_stride_c,
        BLOCK_M, BLOCK_K, BLOCK_N
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)