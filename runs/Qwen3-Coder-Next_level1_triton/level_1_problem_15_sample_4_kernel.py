import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lower_triangular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block offsets
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for valid indices
    mask_m = offset_m < N
    mask_n = offset_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, N, BLOCK_SIZE_K):
        # Compute current K offsets
        current_k = k + offset_k
        mask_k = current_k < N
        
        # Create broadcast masks for the current block
        # For lower triangular: we need m >= k for A[m,k] to be non-zero
        # and k >= n for B[k,n] to be non-zero (but since we're in lower triangle, we need k >= n)
        # Actually for C[m,n] = sum_k A[m,k] * B[k,n], we need:
        # - A[m,k] != 0 only if m >= k
        # - B[k,n] != 0 only if k >= n
        # So C[m,n] != 0 only if there exists k such that m >= k >= n, which requires m >= n
        
        # Load A: only if m >= k
        A_offsets = offset_m[:, None] * stride_am + current_k[None, :] * stride_ak
        A_mask = (mask_m[:, None] & mask_k[None, :] & (offset_m[:, None] >= current_k[None, :]))
        a = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
        
        # Load B: only if k >= n
        B_offsets = current_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
        B_mask = (mask_k[:, None] & mask_n[None, :] & (current_k[:, None] >= offset_n[None, :]))
        b = tl.load(B_ptr + B_offsets, mask=B_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Convert to float16 if needed, but keep as float32 for precision
    acc = acc.to(tl.float32)
    
    # Store only lower triangular part (m >= n)
    C_offsets = offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    mask_lower = (mask_m[:, None] & mask_n[None, :] & (offset_m[:, None] >= offset_n[None, :]))
    tl.store(C_ptr + C_offsets, acc, mask=mask_lower)


def triton_lower_triangular_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication of lower triangular matrices A and B.
    Only computes the lower triangular part of the result.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    assert A.shape == B.shape, "Input matrices must have the same shape"
    assert A.shape[0] == A.shape[1], "Input matrices must be square"
    
    N = A.shape[0]
    
    # Create output tensor
    C = torch.empty_like(A)
    
    # Configure block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    # Grid configuration
    grid = (
        triton.cdiv(N, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    lower_triangular_matmul_kernel[grid](
        A, B, C,
        N,
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
    Optimized model that performs matrix multiplication of lower triangular matrices
    using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_lower_triangular_matmul(A, B)