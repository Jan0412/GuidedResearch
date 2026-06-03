import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_scalar_kernel(
    a_ptr,  # Pointer to input matrix A
    c_ptr,  # Pointer to output matrix C
    m, n,  # Matrix dimensions
    s,  # Scalar value (passed as float32)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Calculate block positions
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    
    # Compute the offsets for rows and columns
    row_offsets = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks to handle boundary conditions
    row_mask = row_offsets < m
    col_mask = col_offsets < n
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Load data from A matrix
    a_block = tl.load(a_ptr + row_offsets[:, None] * n + col_offsets[None, :], mask=mask, other=0.0)
    
    # Multiply by scalar
    c_block = a_block * s
    
    # Store result to C matrix
    tl.store(c_ptr + row_offsets[:, None] * n + col_offsets[None, :], c_block, mask=mask)


def triton_matmul_scalar(a: torch.Tensor, s: float) -> torch.Tensor:
    """
    Performs matrix-scalar multiplication using Triton kernel.
    
    Args:
        a: Input matrix of shape (M, N)
        s: Scalar value
    
    Returns:
        c: Resulting matrix of shape (M, N)
    """
    assert a.is_cuda, "Input tensor must be on CUDA."
    a = a.contiguous()
    
    # Prepare output tensor
    c = torch.empty_like(a)
    
    # Get matrix dimensions
    m, n = a.shape
    
    # Define block sizes (tunable for performance)
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(m, BLOCK_M),
        triton.cdiv(n, BLOCK_N),
    )
    
    # Launch kernel
    matmul_scalar_kernel[grid](a, c, m, n, s, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-scalar multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using optimized Triton kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_matmul_scalar(A, s)