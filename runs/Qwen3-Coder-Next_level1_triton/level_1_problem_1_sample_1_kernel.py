import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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

    # Offsets for pointers
    off_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + off_m * stride_am + off_k * stride_ak
    b_ptrs = B + off_k * stride_bk + off_n * stride_bn

    # Accumulator for the result
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Matrix multiplication loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles from A and B
        a = tl.load(a_ptrs, mask=off_k < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=off_k[:, None] < K - k, other=0.0)
        # Accumulate partial products
        acc += tl.dot(a, b)
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Cast to output type and store result
    c = acc.to(tl.float32)
    c_ptrs = C + off_m * stride_cm + off_n * stride_cn
    mask_m = off_m + tl.arange(0, BLOCK_SIZE_M) < M
    mask_n = off_n + tl.arange(0, BLOCK_SIZE_N) < N
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A @ B using Triton kernel.
    
    Args:
        A (torch.Tensor): Input matrix A of shape (M, K)
        B (torch.Tensor): Input matrix B of shape (K, N)
    
    Returns:
        torch.Tensor: Output matrix C of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Configure kernel parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Grid configuration
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
    Optimized model that performs a single square matrix multiplication (C = A * B)
    using Triton kernel for improved performance.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using Triton kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return triton_matmul(A, B)