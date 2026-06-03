import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mul_scalar_kernel(
    a_ptr,  # Pointer to input matrix A
    out_ptr,  # Pointer to output matrix C
    m, n,  # Dimensions of the matrix
    s,  # Scalar value (passed as a constant)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Compute the starting indices for this block
    m_start = tl.program_id(0) * BLOCK_M
    n_start = tl.program_id(1) * BLOCK_N
    
    # Create ranges for m and n dimensions
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    # Create meshgrid of offsets
    m_idx, n_idx = tl.meshgrid(m_offsets, n_offsets)
    m_idx = tl.reshape(m_idx, [BLOCK_M * BLOCK_N])
    n_idx = tl.reshape(n_idx, [BLOCK_M * BLOCK_N])
    
    # Compute linear offset
    offset = m_idx * n + n_idx
    
    # Create mask to stay within bounds
    mask = (m_idx < m) & (n_idx < n)
    
    # Load data
    a = tl.load(a_ptr + offset, mask=mask, other=0.0)
    
    # Multiply by scalar
    out = a * s
    
    # Store result
    tl.store(out_ptr + offset, out, mask=mask)


def triton_mul_scalar(a: torch.Tensor, s: float) -> torch.Tensor:
    """
    This function wraps the Triton kernel call for matrix-scalar multiplication.
    """
    assert a.is_cuda, "Tensor must be on CUDA."
    a = a.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(a)
    
    # Get dimensions
    m, n = a.shape
    
    # Set block sizes (tunable parameters)
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Calculate grid dimensions
    grid_m = (m + BLOCK_M - 1) // BLOCK_M
    grid_n = (n + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    
    # Launch the Triton kernel
    mul_scalar_kernel[grid](a, out, m, n, s, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out


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
        return triton_mul_scalar(A, s)