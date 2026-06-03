import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_scalar_kernel(
    a_ptr,  # Pointer to input matrix A
    out_ptr,  # Pointer to output matrix
    M,  # Number of rows in A
    N,  # Number of columns in A
    s,  # Scalar multiplier
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Compute the row and column indices for this program instance
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create ranges for row and column offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks to ensure we stay within bounds
    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load the block of A
    a = tl.load(a_ptr + rm[:, None] * N + rn[None, :], mask=mask, other=0.0)
    
    # Multiply by scalar and store result
    out = a * s
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], out, mask=mask)


def triton_matmul_scalar(a: torch.Tensor, s: float):
    """
    Performs matrix-scalar multiplication using Triton kernel.
    
    Args:
        a: Input matrix of shape (M, N)
        s: Scalar value
    
    Returns:
        Resulting matrix of shape (M, N)
    """
    assert a.is_cuda, "Input tensor must be on CUDA."
    a = a.contiguous()
    
    M, N = a.shape
    out = torch.empty_like(a)
    
    # Define block sizes (tunable parameters)
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Compute grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )
    
    # Launch the kernel
    matmul_scalar_kernel[grid](a, out, M, N, s, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-scalar multiplication using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using Triton kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_matmul_scalar(A, s)