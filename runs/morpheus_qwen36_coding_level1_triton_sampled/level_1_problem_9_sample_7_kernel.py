import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < M
    mask_k = offs_k < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        A = tl.load(A_ptr + offs_m[:, None] * N + offs_k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        B = tl.load(B_ptr + offs_k[:, None] * M + offs_n[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(A, B)

    tl.store(C_ptr + offs_m[:, None] * M + offs_n[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()
    M, N = A.shape
    assert B.shape == (N, M), "Shape mismatch"
    C = torch.empty((M, M), dtype=torch.float32, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (M + BLOCK_N - 1) // BLOCK_N

    matmul_kernel[(num_blocks_m, num_blocks_n, 1)](
        A, B, C, M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)