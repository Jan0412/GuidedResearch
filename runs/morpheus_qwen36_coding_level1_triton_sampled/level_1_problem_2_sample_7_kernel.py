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

    rows_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rows_m < M
    mask_n = cols_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        cols_k = k + tl.arange(0, BLOCK_K)
        a_tile = tl.load(A_ptr + rows_m[:, None] * stride_am + cols_k[None, :] * stride_ak,
                         mask=mask_m[:, None] & (cols_k < K), other=0.0)
        b_tile = tl.load(B_ptr + cols_k[:, None] * stride_bk + cols_n[None, :] * stride_bn,
                         mask=(cols_k < K)[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(a_tile, b_tile, acc)

    tl.store(C_ptr + rows_m[:, None] * stride_cm + cols_n[None, :] * stride_cn,
             acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Dimension mismatch: A has K={K}, B has K={K2}"

    C = torch.empty((M, N), dtype=torch.float32, device='cuda')

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N, 1)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)