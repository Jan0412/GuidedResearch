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
    block_m_id = tl.program_id(0)
    block_n_id = tl.program_id(1)

    m_offsets = block_m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = block_n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load A tile
        a_offsets = stride_am * m_offsets[:, None] + stride_ak * (k + k_offsets[None, :])
        mask_a = mask_m[:, None] & ((k + k_offsets[None, :]) < K)
        a = tl.load(A_ptr + a_offsets, mask=mask_a, other=0.0)

        # Load B tile
        b_offsets = stride_bk * (k + k_offsets[:, None]) + stride_bn * n_offsets[None, :]
        mask_b = ((k + k_offsets[:, None]) < K) & mask_n[None, :]
        b = tl.load(B_ptr + b_offsets, mask=mask_b, other=0.0)

        # Accumulate matmul
        acc = tl.matmul(a, b)

    # Store result
    c_offsets = stride_cm * m_offsets[:, None] + stride_cn * n_offsets[None, :]
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), dtype=torch.float32, device='cuda')

    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 256

    num_block_m = (M + BLOCK_M - 1) // BLOCK_M
    num_block_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (num_block_m, num_block_n, 1)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=2
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)