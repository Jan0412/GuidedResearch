import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_3d_kernel(
    A_ptr,  # Pointer to 3D tensor A of shape (N, M, K)
    B_ptr,  # Pointer to matrix B of shape (K, L)
    C_ptr,  # Pointer to output tensor C of shape (N, M, L)
    N, M, K, L,
    stride_a_n, stride_a_m, stride_a_k,
    stride_b_k, stride_b_l,
    stride_c_n, stride_c_m, stride_c_l,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    # Compute batch indices (n_idx, m_idx) for the 3D tensor
    batch_n = tl.program_id(0)
    batch_m = tl.program_id(1)
    
    # Compute the starting offset for this batch in the 3D tensor
    A_offset = batch_n * stride_a_n + batch_m * stride_a_m
    C_offset = batch_n * stride_c_n + batch_m * stride_c_m
    
    # Initialize accumulator for the output
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_L), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        k_end = tl.minimum(k + BLOCK_SIZE_K, K)
        
        # Load block of A: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_offsets = (
            tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_a_m +
            (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * stride_a_k
        )
        a_mask = (
            (tl.arange(0, BLOCK_SIZE_M) < M)[:, None] &
            ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < K)
        )
        a = tl.load(A_ptr + A_offset + a_offsets, mask=a_mask, other=0.0)
        
        # Load block of B: shape (BLOCK_SIZE_K, BLOCK_SIZE_L)
        b_offsets = (
            (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * stride_b_k +
            tl.arange(0, BLOCK_SIZE_L)[None, :] * stride_b_l
        )
        b_mask = (
            ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < K) &
            (tl.arange(0, BLOCK_SIZE_L)[None, :] < L)
        )
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate the matrix multiplication
        accumulator = tl.dot(a, b, accumulator)
    
    # Store the result
    c_offsets = (
        tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_c_m +
        tl.arange(0, BLOCK_SIZE_L)[None, :] * stride_c_l
    )
    c_mask = (
        (tl.arange(0, BLOCK_SIZE_M) < M)[:, None] &
        (tl.arange(0, BLOCK_SIZE_L)[None, :] < L)
    )
    tl.store(C_ptr + C_offset + c_offsets, accumulator, mask=c_mask)


def triton_matmul_3d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication using Triton kernel.
    
    Args:
        A: Input 3D tensor of shape (N, M, K)
        B: Input matrix of shape (K, L)
    
    Returns:
        Output tensor of shape (N, M, L)
    """
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    N, M, K = A.shape
    K2, L = B.shape
    assert K == K2, f"Incompatible dimensions: A has shape {A.shape}, B has shape {B.shape}"
    
    # Create output tensor
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Set up block sizes for performance optimization
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_L = 32
    BLOCK_SIZE_K = 64
    
    # Define grid dimensions: (N, M) for the batch dimensions
    grid = (N, M)
    
    # Compute strides
    stride_a_n = A.stride(0)
    stride_a_m = A.stride(1)
    stride_a_k = A.stride(2)
    stride_b_k = B.stride(0)
    stride_b_l = B.stride(1)
    stride_c_n = C.stride(0)
    stride_c_m = C.stride(1)
    stride_c_l = C.stride(2)
    
    # Launch the kernel
    matmul_3d_kernel[grid](
        A, B, C,
        N, M, K, L,
        stride_a_n, stride_a_m, stride_a_k,
        stride_b_k, stride_b_l,
        stride_c_n, stride_c_m, stride_c_l,
        BLOCK_SIZE_N=1,  # We process one batch element at a time per block
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of the Model class using custom Triton kernel for 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using optimized Triton kernel.
        """
        return triton_matmul_3d(A, B)