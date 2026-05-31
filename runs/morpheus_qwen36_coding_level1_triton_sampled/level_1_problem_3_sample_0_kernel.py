import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    row_tile = tl.program_id(1)
    col_tile = tl.program_id(2)
    
    row_off = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    col_off = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    
    A_ptr = A_ptr + batch * stride_ab
    B_ptr = B_ptr + batch * stride_bk
    C_ptr = C_ptr + batch * stride_cm
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        a_off = k_off + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + row_off[:, None] * stride_ak + a_off[None, :], 
                    mask=(row_off[:, None] < M) & (a_off[None, :] < K), other=0.0)
        b = tl.load(B_ptr + a_off[:, None] * stride_bk + col_off[None, :] * stride_bn, 
                    mask=(a_off[:, None] < K) & (col_off[None, :] < N), other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        
    tl.store(C_ptr + row_off[:, None] * stride_cm + col_off[None, :] * stride_cn, 
             acc, mask=(row_off[:, None] < M) & (col_off[None, :] < N))

def triton_bmm(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    n = B.shape[2]
    C = torch.empty(batch_size, m, n, dtype=A.dtype, device=A.device)
    
    BLOCK_M = 128
    BLOCK_K = 64
    BLOCK_N = 128
    
    num_row_tiles = (m + BLOCK_M - 1) // BLOCK_M
    num_col_tiles = (n + BLOCK_N - 1) // BLOCK_N
    
    grid = (batch_size, num_row_tiles, num_col_tiles)
    
    bmm_kernel[grid](
        A, B, C,
        m, n, k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N
    )
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)