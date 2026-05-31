import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a_mask = mask_m[:, None] & (k + offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + (k + offs_k[:, None]) * stride_bn + offs_n[None, :] * stride_bk
        b_mask = (k + offs_k[:, None] < K) & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    
    batch_size, m, k = A.shape
    _, _, n = B.shape
    C = torch.empty((batch_size, m, n), dtype=A.dtype, device=A.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    for i in range(batch_size):
        grid = ((m + BLOCK_M - 1) // BLOCK_M, (n + BLOCK_N - 1) // BLOCK_N)
        bmm_kernel[grid](
            A[i], B[i], C[i],
            m, n, k,
            A.stride(1), A.stride(2),
            B.stride(1), B.stride(2),
            C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)


def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]


def get_init_inputs():
    return []