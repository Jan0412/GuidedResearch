import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tril_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row and column indices for this block
    row_start = tl.program_id(0)
    col_start = tl.program_id(1)
    
    # Create offsets for rows and columns
    row_offsets = row_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = col_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid rows and columns
    row_mask = row_offsets < N
    col_mask = col_offsets < N
    
    # Initialize output accumulator
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # For each row, we only need to compute up to the diagonal (column index = row index)
    # So we process columns in chunks, but only accumulate when col <= row
    for k in range(0, N, BLOCK_SIZE):
        # Create k offsets
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < N
        
        # Load A[row, k] - A is lower triangular so we need row >= k
        a_tile = tl.load(A_ptr + row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B[k, col] - B is lower triangular so we need k >= col
        b_tile = tl.load(B_ptr + k_offsets[:, None] * stride_bk + col_offsets[None, :] * stride_bn,
                        mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        
        # For lower triangular matrices:
        # C[i,j] only depends on A[i,k] * B[k,j] where k <= min(i,j)
        # Since we're computing for row i and col j, we need k <= min(i,j)
        # But we're doing batch computation, so we need to mask appropriately
        
        # Compute partial product
        partial = tl.dot(a_tile, b_tile, out_dtype=tl.float32)
        
        # Only accumulate where row >= col (lower triangular part)
        row_idx = row_offsets[:, None]
        col_idx = col_offsets[None, :]
        valid = (row_idx >= col_idx) & row_mask[:, None] & col_mask[None, :]
        
        output = tl.where(valid, output + partial, output)
    
    # Store the result
    tl.store(C_ptr + row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn,
             output.to(tl.float32), mask=row_mask[:, None] & col_mask[None, :])


def triton_tril_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized matrix multiplication for lower triangular matrices.
    Computes C = tril(A @ B) where A and B are lower triangular.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    
    # Create output tensor
    C = torch.empty_like(A)
    
    # Set block size - tune based on GPU
    BLOCK_SIZE = 64
    
    # Grid: one block per BLOCK_SIZE x BLOCK_SIZE tile
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, 
            (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch kernel
    tril_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model for lower triangular matrix multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        return triton_tril_matmul(A, B)