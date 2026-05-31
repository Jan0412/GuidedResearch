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
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    off_m = pid_m * BLOCK_SIZE_M
    off_n = pid_n * BLOCK_SIZE_N

    # Pointer arithmetic
    off_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (off_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_am + off_k[None, :] * stride_ak
    b_ptrs = B_ptr + off_k[:, None] * stride_bk + (off_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_bn

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks
        a = tl.load(a_ptrs, mask=off_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=off_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Matrix multiply
        accumulator += tl.dot(a, b)
        # Update pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    c = accumulator.to(tl.float32)
    c_ptrs = C_ptr + (off_m + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_cm + (off_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_cn
    mask = (off_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (off_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_matmul_transposed(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T @ B^T using custom Triton kernel.
    Note: A is (K, M), B is (N, K), so A^T is (M, K), B^T is (K, N), result is (M, N).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Ensure correct shapes
    assert A.shape[1] == B.shape[1], "A.shape[1] must equal B.shape[1] (both K)"
    K, M = A.shape
    N, _ = B.shape

    # Prepare output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Set block sizes (tunable)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Grid dimensions
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
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
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication C = A^T @ B^T using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using custom Triton kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transposed(A, B)