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
    """
    Triton kernel for matrix multiplication C = A @ B
    A: (M, K), B: (K, N), C: (M, N)
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # Mask for valid indices
    mask_am = offsets_am < M
    mask_bn = offsets_bn < N
    mask_k = offsets_k < K

    # Load blocks of A and B
    A = tl.load(A_ptr + offsets_am[:, None] * stride_am + offsets_k[None, :] * stride_ak, 
                mask=mask_am[:, None] & mask_k[None, :], other=0.0)
    B = tl.load(B_ptr + offsets_k[:, None] * stride_bk + offsets_bn[None, :] * stride_bn, 
                mask=mask_k[:, None] & mask_bn[None, :], other=0.0)

    # Compute partial dot product
    C = tl.dot(A, B)

    # Store result
    offsets_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cm = offsets_cm < M
    mask_cn = offsets_cn < N

    tl.store(C_ptr + offsets_cm[:, None] * stride_cm + offsets_cn[None, :] * stride_cn, 
             C, mask=mask_cm[:, None] & mask_cn[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for Triton matmul kernel.
    Computes C = A @ B where A is (M, K) and B is (K, N).
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must match."

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Grid configuration
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128), 1)

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
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
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).
        Returns:
            Output tensor of shape (M, N).
        """
        # Compute A @ B.T using Triton matmul
        return triton_matmul(A, B)