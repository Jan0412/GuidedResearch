import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr,  # Pointer to upper triangular matrix A
    B_ptr,  # Pointer to upper triangular matrix B
    C_ptr,  # Pointer to output matrix C
    N,      # Dimension of the matrices (N x N)
    stride_a, stride_b, stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs for row and column blocks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the starting row and column indices for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Create row and column offsets
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Create broadcasted meshes for row and column indices
    row_indices = row_offsets[:, None]
    col_indices = col_offsets[None, :]
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension in blocks
    for k_start in range(0, N, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        
        # Calculate which elements of K dimension are valid for this (row, col) pair
        # For upper triangular matrices: A[i, k] != 0 only if i <= k
        #                                B[k, j] != 0 only if k <= j
        # So the valid range is max(i, k_start) <= k <= min(j, k_start + BLOCK_SIZE_K - 1)
        
        # Load A block: A[row, k] where row in [row_start, row_start+BLOCK_SIZE_M) and k in [k_start, k_start+BLOCK_SIZE_K)
        a_mask = (row_indices < N)[:, None] & (k_offsets[None, :] < N)[None, :]
        a_ptrs = A_ptr + row_indices * stride_a + k_offsets[None, :] * stride_a
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block: B[k, col] where k in [k_start, k_start+BLOCK_SIZE_K) and col in [col_start, col_start+BLOCK_SIZE_N)
        b_mask = (k_offsets[:, None] < N) & (col_indices[None, :] < N)
        b_ptrs = B_ptr + k_offsets[:, None] * stride_b + col_indices[None, :] * stride_b
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate multiplication only for valid upper triangular positions
        # For position (i,j), we only add to accumulator if i <= j
        # We'll mask the final result later, but for efficiency, we can compute partial products
        
        # Matrix multiply: accumulator += A @ B
        accumulator += tl.dot(a, b)
    
    # Store result only for upper triangular positions
    output_mask = (row_indices < N) & (col_indices < N) & (row_indices <= col_indices)
    c_ptrs = C_ptr + row_indices * stride_c + col_indices * stride_c
    tl.store(c_ptrs, accumulator, mask=output_mask)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication for upper triangular matrices using Triton kernel.
    Only computes the upper triangular part of the result.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices A and B must have the same shape."
    assert len(A.shape) == 2, "Matrices must be 2D."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    N = A.shape[0]
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Prepare output tensor
    C = torch.empty_like(A)
    
    # Strides for memory access
    stride_a = A.stride(0)
    stride_b = B.stride(0)
    stride_c = C.stride(0)
    
    # Tunable parameters for block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
    grid = (
        triton.cdiv(N, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch the kernel
    triu_matmul_kernel[grid](
        A, B, C, N,
        stride_a, stride_b, stride_c,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using Triton kernel.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)