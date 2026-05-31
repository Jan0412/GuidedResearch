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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_blocks = (K + BLOCK_K - 1) // BLOCK_K
    for k in range(num_k_blocks):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k * BLOCK_K + offs_k[None, :]) * stride_ak
        b_ptrs = B_ptr + (k * BLOCK_K + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        A = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        B = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(A, B)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible dimensions: {A.shape} and {B.shape}"

    C = torch.empty((M, N), dtype=torch.float32, device='cuda')

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
        1
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)