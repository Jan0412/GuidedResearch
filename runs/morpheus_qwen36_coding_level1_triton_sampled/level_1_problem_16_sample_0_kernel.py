import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K, N,
    stride_A, stride_B, stride_C,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr
):
    pid = tl.program_id(0)
    num_cols = triton.cdiv(N, BLOCK_N)
    block_n = pid % num_cols
    block_m = pid // num_cols
    
    off_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    
    mask_m = off_m < M
    mask_n = off_n < N
    mask_k = off_k < K
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        A_tile = tl.load(A_ptr + off_m[:, None] * stride_A + off_k[None, :], 
                         mask=mask_m[:, None] & mask_k[None, :], 
                         other=0.0)
        B_tile = tl.load(B_ptr + off_k[:, None] * stride_B + off_n[None, :], 
                         mask=mask_k[:, None] & mask_n[None, :], 
                         other=0.0)
        acc += tl.dot(A_tile, B_tile)
        
    off_c = off_m[:, None] * stride_C + off_n[None, :]
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + off_c, acc, mask=mask_c)


def triton_matmul(A, B):
    A = A.contiguous()
    B = B.contiguous()
    
    M = A.shape[1]
    K = A.shape[0]
    N = B.shape[1]
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = lambda meta: (triton.cdiv(N, BLOCK_N) * triton.cdiv(M, BLOCK_M),)
    
    matmul_kernel[grid](
        A, B, C,
        M, K, N,
        M, N, N,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_stages=3
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)