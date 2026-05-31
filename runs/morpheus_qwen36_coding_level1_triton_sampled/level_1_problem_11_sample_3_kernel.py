import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    mask_m = offsets_am < M
    mask_n = offsets_bn < N

    a_ptrs = A_ptr + offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = B_ptr + offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (offsets_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :] & (offsets_k[:, None] < K - k), other=0.0)
        accumulator = tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offsets_am[:, None] * stride_cm + offsets_bn[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, accumulator, mask=mask_c)


def triton_gemm(A, B):
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    b, i, j, l = A.shape
    k = B.shape[1]
    M = b * i * j
    N = k
    K = l

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N
    )

    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    return C.view(b, i, j, k)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_gemm(A, B)