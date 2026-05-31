import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A [M, K]
    B_ptr,  # Pointer to vector B [K, 1]
    C_ptr,  # Pointer to output C [M, 1]
    M, K,   # Dimensions
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID for M dimension
    pid_m = tl.program_id(0)
    
    # Create offsets for M dimension
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < M
    
    # Initialize accumulator for dot product
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load block of A: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_block = tl.load(
            A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Load block of B: [BLOCK_SIZE_K, 1]
        b_block = tl.load(
            B_ptr + k_offsets[:, None] * stride_bk,
            mask=k_mask[:, None],
            other=0.0
        )
        
        # Perform matrix-vector multiplication for this block
        accumulator += tl.sum(a_block * b_block, axis=1)
    
    # Store result to C
    c_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_mask = c_offsets < M
    tl.store(
        C_ptr + c_offsets[:, None] * stride_cm,
        accumulator[:, None],
        mask=c_mask[:, None]
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector multiplication using Triton kernel.
    A: [M, K], B: [K, 1] -> C: [M, 1]
    """
    # Ensure inputs are contiguous and on CUDA
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Get dimensions
    M, K = A.shape
    K_b, N = B.shape
    
    # Check dimension compatibility
    assert K == K_b, f"Matrix dimensions not compatible: A[{M}, {K}], B[{K_b}, {N}]"
    
    # Prepare output tensor
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # Define block sizes for optimization
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 256
    BLOCK_SIZE_N = 1
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-vector multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernel.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matmul(A, B)