import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_3d_kernel(
    A_ptr,  # Pointer to 3D tensor A with shape (N, M, K)
    B_ptr,  # Pointer to matrix B with shape (K, L)
    C_ptr,  # Pointer to output tensor with shape (N, M, L)
    N, M, K, L,
    stride_a0, stride_a1, stride_a2,  # Strides for A: (M*K, K, 1)
    stride_b0, stride_b1,            # Strides for B: (L, 1)
    stride_c0, stride_c1,            # Strides for C: (M*L, L)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    """
    Triton kernel for 3D tensor-matrix multiplication: A (N, M, K) @ B (K, L) -> C (N, M, L)
    
    Each program handles one (i, j) pair where i is in [0, N) and j is in [0, M),
    computing the matrix-vector product for the vector A[i, j, :] with matrix B[:, :].
    """
    # Program ID encodes both batch index (n_idx) and row index (m_idx)
    # We use grid flattening: program_id = n_idx * M + m_idx
    n_idx = tl.program_id(0) // M
    m_idx = tl.program_id(0) % M
    
    # Check bounds
    if n_idx >= N or m_idx >= M:
        return
    
    # Offset pointers to the correct slice A[n_idx, m_idx, :]
    a_offset = n_idx * stride_a0 + m_idx * stride_a1
    c_offset = n_idx * stride_c0 + m_idx * stride_c1
    
    # Initialize accumulator for the result row (1 × L)
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_end = tl.minimum(k_start + BLOCK_SIZE_K, K)
        k_range = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_range < K
        
        # Load a block of A: shape (BLOCK_SIZE_K,)
        a_block = tl.load(A_ptr + a_offset + k_range * stride_a2, mask=k_mask)
        
        # Load a block of B: shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # We'll load columns of B in blocks
        b_block = tl.load(
            B_ptr + k_range[:, None] * stride_b0 + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_b1,
            mask=k_mask[:, None] & (tl.arange(0, BLOCK_SIZE_N) < L)
        )
        
        # Accumulate: a_block (BLOCK_SIZE_K,) @ b_block (BLOCK_SIZE_K, BLOCK_SIZE_N)
        accumulator += tl.sum(a_block[:, None] * b_block, axis=0)
    
    # Store the result row to C
    c_row = accumulator.to(tl.float32)
    c_mask = tl.arange(0, BLOCK_SIZE_N) < L
    tl.store(
        C_ptr + c_offset + tl.arange(0, BLOCK_SIZE_N) * stride_c1,
        c_row,
        mask=c_mask
    )


def triton_matmul_3d(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication using Triton kernel.
    
    Args:
        A (torch.Tensor): Input 3D tensor of shape (N, M, K).
        B (torch.Tensor): Input matrix of shape (K, L).
    
    Returns:
        torch.Tensor: Output tensor of shape (N, M, L).
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    
    # Validate shapes
    assert len(A.shape) == 3, f"A must be 3D, got shape {A.shape}"
    assert len(B.shape) == 2, f"B must be 2D, got shape {B.shape}"
    assert A.shape[2] == B.shape[0], f"Inner dimensions must match: A.shape={A.shape}, B.shape={B.shape}"
    
    N, M, K = A.shape
    _, L = B.shape
    
    # Prepare output tensor
    C = torch.empty((N, M, L), dtype=A.dtype, device=A.device)
    
    # Configure kernel parameters
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = 64  # Tile size for L dimension
    BLOCK_SIZE_K = 32  # Tile size for K dimension
    
    # Calculate grid size: N * M programs (one per row in the batch)
    grid = (N * M,)
    
    # Launch kernel
    matmul_3d_kernel[grid](
        A, B, C,
        N, M, K, L,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using optimized Triton kernel.
        """
        return triton_matmul_3d(A, B)