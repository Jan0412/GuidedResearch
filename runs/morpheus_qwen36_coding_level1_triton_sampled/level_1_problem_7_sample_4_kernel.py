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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_am = offs_am < M
    mask_bn = offs_bn < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        offs_ak = k + tl.arange(0, BLOCK_SIZE_K)
        offs_bk = k + tl.arange(0, BLOCK_SIZE_K)
        
        a_block = tl.load(
            A_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak,
            mask=mask_am[:, None] & (offs_ak < K),
            other=0.0
        )
        b_block = tl.load(
            B_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_bk < K)[:, None] & mask_bn[None, :],
            other=0.0
        )
        
        acc += tl.dot(a_block, b_block)
        
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cm = mask_am[:, None] & mask_bn[None, :]
    
    tl.store(
        C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
        acc,
        mask=mask_cm
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Incompatible dimensions"
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)