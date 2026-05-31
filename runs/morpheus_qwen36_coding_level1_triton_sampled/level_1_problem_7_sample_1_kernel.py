import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    for k in range(num_k_blocks):
        mask_k = tl.arange(0, BLOCK_K) < K - k * BLOCK_K
        A = tl.load(A_ptrs, mask=mask_k[None, :], other=0.0)
        B = tl.load(B_ptrs, mask=mask_k[:, None], other=0.0)
        accumulator = tl.dot(A, B, accumulator)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptrs, accumulator, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, 1)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)