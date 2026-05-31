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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_k = tl.arange(0, BLOCK_SIZE_K)

    # Masks
    mask_m = off_m < M
    mask_n = off_n < N
    mask_k = off_k < K

    # Base pointers for current block
    a_ptrs = A_ptr + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    b_ptrs = B_ptr + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load sub-matrices
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

        # Perform dot product
        accumulator = tl.dot(a, b, accumulator)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    c_ptrs = C_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, accumulator, mask=mask_c)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Tunable block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid calculation
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return C


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton matrix multiplication kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.
        Input A: (K, M), Input B: (N, K)
        Computes A.T @ B.T -> (M, K) @ (K, N) -> (M, N)
        """
        # Transpose inputs to match kernel expectations: (M, K) and (K, N)
        # .T creates a view, so this is efficient
        A_T = A.T
        B_T = B.T

        # Call Triton matmul
        return triton_matmul(A_T, B_T)