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
    block_idx_m = tl.program_id(0)
    block_idx_n = tl.program_id(1)

    offs_am = block_idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = block_idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        mask_a = (offs_am[:, None] < M) & ((k + offs_k)[None, :] < K)
        mask_b = ((k + offs_k)[:, None] < K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = torch.empty((M, N), dtype=A.dtype, device='cuda')

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    num_block_m = (M + BLOCK_M - 1) // BLOCK_M
    num_block_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_block_m, num_block_n, 1)

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