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
    # Grid coordinates for this program
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block start indices
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)

    # Create masks for valid indices
    mask_m = off_m < M
    mask_n = off_n < N
    mask_k = off_k < K

    # Pointers to current blocks
    a_ptrs = A_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    b_ptrs = B_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn

    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles from A and B
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

        # Matrix multiply
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert to float16/bfloat16 if needed, otherwise keep as float32
    c = accumulator

    # Store result
    off_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + off_cm[:, None] * stride_cm + off_cn[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only FP32 supported."
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matmul."

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "K dimensions must match."

    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((M, N), dtype=torch.float32, device="cuda")

    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    # Grid calculation
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)