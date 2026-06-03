import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    a_ptr,  # Pointer to input matrix A
    out_ptr,  # Pointer to output matrix
    m, n,  # Dimensions of the matrix
    s,  # Scalar multiplier
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Compute the block index in the M dimension
    pid_m = tl.program_id(0)
    # Compute the block index in the N dimension
    pid_n = tl.program_id(1)
    
    # Compute the starting offsets for this block
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create a mask to ensure we don't go out of bounds
    mask_m = offsets_m < m
    mask_n = offsets_n < n
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load the block of A
    a = tl.load(a_ptr + offsets_m[:, None] * n + offsets_n[None, :], mask=mask, other=0.0)
    
    # Scale by the scalar
    out = a * s
    
    # Store the result
    tl.store(out_ptr + offsets_m[:, None] * n + offsets_n[None, :], out, mask=mask)


def triton_scale(a: torch.Tensor, s: float):
    """
    This function wraps the Triton kernel call for matrix-scalar multiplication.
    
    Args:
        a: Input tensor of shape (M, N)
        s: Scalar value to multiply with
    
    Returns:
        Result tensor of shape (M, N)
    """
    assert a.is_cuda, "Tensor must be on CUDA."
    a = a.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(a)
    
    # Dimensions
    m, n = a.shape
    
    # Block sizes - chosen for good occupancy and memory utilization
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Grid size: number of blocks needed in each dimension
    grid = (
        triton.cdiv(m, BLOCK_M),
        triton.cdiv(n, BLOCK_N),
    )
    
    # Launch the kernel
    scale_kernel[grid](a, out, m, n, s, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    
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
        return triton_scale(A, s)