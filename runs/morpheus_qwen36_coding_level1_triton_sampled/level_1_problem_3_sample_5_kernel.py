import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, k, n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < m
    mask_n = offs_n < n

    A_ptr += pid_b * m * k + offs_m[:, None] * k
    B_ptr += pid_b * k * n + offs_n[None, :]
    C_ptr += pid_b * m * n + offs_m[:, None] * n + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, k, BLOCK_K):
        mask_k = offs_k < k
        A_tile = tl.load(A_ptr + offs_k[None, :], mask=mask_k, other=0.0)
        B_tile = tl.load(B_ptr + offs_k[:, None] * n, mask=mask_k, other=0.0)
        acc = tl.dot(A_tile, B_tile, acc=acc)

    tl.store(C_ptr, acc, mask=mask_n)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()

    batch_size, m, k = A.shape
    _, _, n = B.shape

    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda META: (
        batch_size,
        triton.cdiv(m, META["BLOCK_M"]),
        triton.cdiv(n, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        A, B, C,
        batch_size, m, k, n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)