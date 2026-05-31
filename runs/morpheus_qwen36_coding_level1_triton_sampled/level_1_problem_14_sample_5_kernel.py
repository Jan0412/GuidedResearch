import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Upper triangular mask (includes diagonal to match torch.triu default)
    mask_upper = m_offsets[:, None] <= n_offsets[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # Boundary masks for A and B
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        a = tl.load(A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(B_ptr + k_offsets[:, None] * stride_bn + n_offsets[None, :] * stride_cn, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32)

    # Only store results for the upper triangular part
    c_mask = mask_upper & (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn, acc, mask=c_mask)


def triton_triu_matmul(A, B):
    assert A.is_cuda and B.is_cuda
    A = A.contiguous()
    B = B.contiguous()
    N = A.shape[0]
    K = A.shape[1]
    assert A.shape[1] == B.shape[0]

    C = torch.empty_like(A)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = ((N + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    triu_matmul_kernel[grid](
        A, B, C,
        N, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_triu_matmul(A, B)


N = 4096

def get_inputs():
    A = torch.triu(torch.rand(N, N))
    B = torch.triu(torch.rand(N, N))
    return [A, B]

def get_init_inputs():
    return []