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

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < K - k
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        accumulator = tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = offs_am[:, None]
    offs_cn = offs_bn[None, :]
    c_ptrs = C_ptr + offs_cm * stride_cm + offs_cn * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "A's columns must match B's rows"

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
        1
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B) 
    using a custom tiled Triton kernel for FP32 precision.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)