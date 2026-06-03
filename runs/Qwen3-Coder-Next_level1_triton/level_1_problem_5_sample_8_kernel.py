import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    a_ptr,  # Pointer to input matrix A
    out_ptr,  # Pointer to output matrix C
    m, n,  # Dimensions of the matrix
    s,  # Scalar value to multiply with
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the starting offsets for this block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks to handle boundary conditions
    mask_m = rm < m
    mask_n = rn < n
    
    # Create 2D mask for the block
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load the block from input matrix A
    a_ptrs = a_ptr + rm[:, None] * n + rn[None, :]
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    
    # Scale the values
    out = a * s
    
    # Store the result to output matrix C
    out_ptrs = out_ptr + rm[:, None] * n + rn[None, :]
    tl.store(out_ptrs, out, mask=mask)


def triton_scale(a: torch.Tensor, s: float) -> torch.Tensor:
    """
    This function wraps the Triton kernel call for matrix-scalar multiplication.
    
    Args:
        a: Input matrix of shape (M, N)
        s: Scalar value to multiply with
    
    Returns:
        Resulting matrix of shape (M, N)
    """
    assert a.is_cuda, "Input tensor must be on CUDA."
    a = a.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(a)
    
    # Get dimensions
    m, n = a.shape
    
    # Define block sizes (tunable parameters)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Compute grid dimensions
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch the Triton kernel
    scale_kernel[grid_m, grid_n](
        a, out, m, n, s,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
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
        return triton_scale(A, s)