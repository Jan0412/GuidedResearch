import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_T_ptr,  # Pointer to A^T (K x M)
    B_T_ptr,  # Pointer to B^T (N x K)
    C_ptr,    # Pointer to output C (M x N)
    M, N, K,
    stride_a0, stride_a1,  # Strides for A^T (K, M)
    stride_b0, stride_b1,  # Strides for B^T (N, K)
    stride_c0, stride_c1,  # Strides for C (M, N)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program IDs for M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block start offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator for C matrix
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute offsets for K dimension
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        
        # Load block from A^T (K x M): shape (BLOCK_SIZE_K, BLOCK_SIZE_M)
        # A^T has shape (K, M), so we use [offs_k, offs_m]
        a = tl.load(
            A_T_ptr + offs_k[:, None] * stride_a0 + offs_m[None, :] * stride_a1,
            mask=(offs_k[:, None] < K) & mask_m[None, :],
            other=0.0
        )
        
        # Load block from B^T (N x K): shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
        # B^T has shape (N, K), so we use [offs_n, offs_k]
        b = tl.load(
            B_T_ptr + offs_n[:, None] * stride_b0 + offs_k[None, :] * stride_b1,
            mask=(offs_n[:, None] < N) & mask_k[None, :],
            other=0.0
        )
        
        # Perform matrix multiplication: (BLOCK_SIZE_K x BLOCK_SIZE_M)^T @ (BLOCK_SIZE_N x BLOCK_SIZE_K)^T
        # = (BLOCK_SIZE_M x BLOCK_SIZE_K) @ (BLOCK_SIZE_K x BLOCK_SIZE_N)
        # = (BLOCK_SIZE_M x BLOCK_SIZE_N)
        acc += tl.dot(a, b)
    
    # Store result to C matrix (M x N)
    c = acc.to(tl.float32)  # Keep as float32 for output
    tl.store(
        C_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1,
        c,
        mask=mask_m[:, None] & mask_n[None, :]
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication C = A^T @ B^T using Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    K, M = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    
    # Create output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
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
        Performs matrix multiplication using Triton kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)