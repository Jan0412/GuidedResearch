import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triu_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row and column indices for this block
    row_start = tl.program_id(0) * BLOCK_SIZE
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # Create ranges for row and column indices
    row_indices = row_start + tl.arange(0, BLOCK_SIZE)
    col_indices = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for upper triangular constraint (row <= col)
    mask_row = row_indices < N
    mask_col = col_indices < N
    mask = mask_row[:, None] & mask_col[None, :] & (row_indices[:, None] <= col_indices[None, :])
    
    # Initialize accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over the k dimension (columns of A / rows of B)
    for k in range(0, N, BLOCK_SIZE):
        # Create k indices
        k_indices = k + tl.arange(0, BLOCK_SIZE)
        mask_k = k_indices < N
        
        # Load A block: A[row, k] - only valid when row <= k
        a_offsets = row_indices[:, None] * stride_am + k_indices[None, :] * stride_an
        a_mask = (row_indices[:, None] < N) & (k_indices[None, :] < N) & (row_indices[:, None] <= k_indices[None, :])
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B block: B[k, col] - only valid when k <= col
        b_offsets = k_indices[:, None] * stride_bm + col_indices[None, :] * stride_bn
        b_mask = (k_indices[:, None] < N) & (col_indices[None, :] < N) & (k_indices[:, None] <= col_indices[None, :])
        b = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Accumulate the product where both A and B are valid (row <= k <= col)
        # Only accumulate when row <= col (upper triangular constraint)
        block_mask = (row_indices[:, None] <= col_indices[None, :])
        acc += tl.dot(a, b, out_dtype=tl.float32) * block_mask
    
    # Store result - only for upper triangular elements
    c_offsets = row_indices[:, None] * stride_cm + col_indices[None, :] * stride_cn
    tl.store(C_ptr + c_offsets, acc.to(tl.float32), mask=mask)


def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs matrix multiplication for upper triangular matrices using Triton.
    Only computes the upper triangular part of the result.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.shape == B.shape, "Matrices must have the same shape."
    assert A.shape[0] == A.shape[1], "Matrices must be square."
    
    A = A.contiguous()
    B = B.contiguous()
    
    N = A.shape[0]
    C = torch.empty_like(A)
    
    # Use a reasonable block size (tunable)
    BLOCK_SIZE = 128
    
    # Grid dimensions: each block computes a BLOCK_SIZE x BLOCK_SIZE tile
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch the kernel
    triu_matmul_kernel[grid](
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
    Optimized model that performs matrix multiplication for upper triangular matrices
    using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)