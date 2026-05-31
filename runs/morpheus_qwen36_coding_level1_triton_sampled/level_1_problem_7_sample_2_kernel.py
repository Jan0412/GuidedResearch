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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    for k in range(num_k_blocks):
        mask_a = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        mask_b = offs_k[:, None] < K - k * BLOCK_K
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = offs_am[:, None]
    offs_cn = offs_bn[None, :]
    c_ptrs = C_ptr + offs_cm * stride_cm + offs_cn * stride_cn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    def grid(meta):
        num_m = (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
        num_n = (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
        return (num_m, num_n, 1)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        K, 1,  # A strides
        N, 1,  # B strides
        N, 1,  # C strides
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)