import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_T_ptr,  # Pointer to A^T (M, K) - note: A is stored as (K, M) but we access as (M, K)
    B_ptr,    # Pointer to B (K, N)
    C_ptr,    # Pointer to output C (M, N)
    M, N, K,
    stride_a0, stride_a1,  # Strides for A^T: (K, M) in memory but accessed as (M, K)
    stride_b0, stride_b1,  # Strides for B: (K, N)
    stride_c0, stride_c1,  # Strides for C: (M, N)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block offsets for M and N dimensions
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to handle boundary conditions
    rm_mask = rm < M
    rn_mask = rn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate the current K offset
        k_offset = k * BLOCK_SIZE_K
        
        # Load block from A^T (which is stored as A in (K, M) format)
        # A^T[m, k] = A[k, m], so we access A_T_ptr[k_offset + tl.arange(0, BLOCK_SIZE_K), rm]
        # But since A_T_ptr is actually A in memory (K, M), we need to transpose the indexing
        a_offsets = k_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_a0 + rm[None, :] * stride_a1
        a_mask = (k_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & rm_mask[None, :]
        a = tl.load(A_T_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load block from B (K, N)
        b_offsets = k_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_b0 + rn[None, :] * stride_b1
        b_mask = (k_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & rn_mask[None, :]
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Compute partial matrix multiplication
        # acc += tl.dot(a, b)  # This would be for A^T @ B
        # But note: a is (BLOCK_SIZE_K, BLOCK_SIZE_M), b is (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # We need to transpose a to get (BLOCK_SIZE_M, BLOCK_SIZE_K) for proper dot product
        acc += tl.dot(a.T, b)
    
    # Store result
    c_offsets = rm[:, None] * stride_c0 + rn[None, :] * stride_c1
    c_mask = rm_mask[:, None] & rn_mask[None, :]
    tl.store(C_ptr + c_offsets, acc.to(C_ptr.type.element_dtype), mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = A^T @ B using custom Triton kernel.
    
    Args:
        A: Input tensor of shape (K, M)
        B: Input tensor of shape (K, N)
    
    Returns:
        Output tensor of shape (M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    K, M = A.shape
    _, N = B.shape
    
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
    
    # Calculate strides
    stride_a0, stride_a1 = A.stride()
    stride_b0, stride_b1 = B.stride()
    stride_c0, stride_c1 = C.stride()
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_a0, stride_a1,
        stride_b0, stride_b1,
        stride_c0, stride_c1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernel.
    Computes C = A^T * B where A has shape (K, M) and B has shape (K, N).
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using optimized Triton kernel.
        
        Args:
            A: Input tensor of shape (M, K) - actually (K, M) in memory
            B: Input tensor of shape (K, N)
        
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)