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

    mask_am = offs_am < M
    mask_bn = offs_bn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, (K + BLOCK_K - 1) // BLOCK_K):
        offs_ak = k * BLOCK_K + offs_k
        offs_bk = k * BLOCK_K + offs_k

        a = tl.load(A_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak,
                    mask=mask_am[:, None] & (offs_ak[None, :] < K), other=0.0)
        b = tl.load(B_ptr + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                    mask=(offs_bk[:, None] < K) & mask_bn[None, :], other=0.0)

        acc += tl.dot(a, b)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = mask_am[:, None] & mask_bn[None, :]
    tl.store(C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    # Compute A.T @ B. Transpose A to make it contiguous with shape (M, K).
    A_T = A.T.contiguous()
    M, K = A_T.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions for matrix multiplication."

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    matmul_kernel[grid](
        A_T, B, C,
        M, N, K,
        A_T.stride(0), A_T.stride(1),
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