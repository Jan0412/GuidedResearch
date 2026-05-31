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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)

    mask_m = off_m < M
    mask_n = off_n < N
    mask_k = off_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load block of A: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + off_m[None, :] * stride_am + (k + off_k)[:, None] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[None, :] & mask_k[:, None], other=0.0)

        # Load block of B: shape (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + (k + off_k)[None, :] * stride_bk + off_n[:, None] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[None, :] & mask_n[:, None], other=0.0)

        # Matrix multiplication
        acc = tl.dot(a, b, acc=acc)

    # Store result
    c_ptrs = C_ptr + off_m[None, :] * stride_cm + off_n[:, None] * stride_cn
    mask_c = mask_m[None, :] & mask_n[:, None]
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    N, _ = B.shape
    
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
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