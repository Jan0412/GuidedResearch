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
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K - k)
        mask_b = (offs_k[:, None] < K - k) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator = tl.dot(a, b, accumulator=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must match"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "FP32 precision required"

    C = torch.empty((M, N), dtype=torch.float32, device='cuda')

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

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