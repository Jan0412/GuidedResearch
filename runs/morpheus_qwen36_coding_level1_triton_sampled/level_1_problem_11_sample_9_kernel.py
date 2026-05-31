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

    offsets_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    mask_am = offsets_am < M
    mask_bn = offsets_bn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offsets_ak = k + offsets_k
        mask_k = offsets_ak < K

        a_ptrs = A_ptr + offsets_am[:, None] * stride_am + offsets_ak[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_am[:, None] & mask_k[None, :], other=0.0)

        b_ptrs = B_ptr + offsets_ak[:, None] * stride_bk + offsets_bn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_bn[None, :], other=0.0)

        acc += tl.dot(a, b)

    offsets_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn
    mask_cm = offsets_cm < M
    mask_cn = offsets_cn < N
    tl.store(c_ptrs, acc, mask=mask_cm[:, None] & mask_cn[None, :])


def triton_matmul(A, B):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    
    b, i, j, l = A.shape
    k = B.shape[1]
    M = b * i * j
    N = k
    K_dim = l

    C = torch.empty(b, i, j, k, device=A.device, dtype=A.dtype)

    stride_am = A.stride(0) * i * j + A.stride(1) * i + A.stride(2)
    stride_ak = A.stride(3)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0) * i * j + C.stride(1) * i + C.stride(2)
    stride_cn = C.stride(3)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K_dim,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)