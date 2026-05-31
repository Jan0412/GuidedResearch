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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication C = A @ B.
    """
    # Block indices
    block_idx_m = tl.program_id(0)
    block_idx_n = tl.program_id(1)

    # Create block coordinates
    offs_m = block_idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load blocks from A and B with masking
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak + k,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn + k,
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Perform dot product
        acc += tl.dot(a, b)

    # Store result
    offs_cm = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C + offs_cm, acc, mask=mask_c)


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
        assert A.shape[1] == B.shape[0], "Incompatible dimensions for matmul."

        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "K dimensions must match."

        # Output tensor
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        # Block sizes (tunable parameters)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid configuration
        grid = lambda meta: (
            (M + BLOCK_M - 1) // BLOCK_M,
            (N + BLOCK_N - 1) // BLOCK_N,
        )

        # Launch kernel
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K,
        )

        return C