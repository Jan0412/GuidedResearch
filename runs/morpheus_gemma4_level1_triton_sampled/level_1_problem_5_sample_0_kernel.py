import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scalar_mul_kernel(
    A_ptr,      # Pointer to input matrix
    out_ptr,    # Pointer to output matrix
    s,          # Scalar value
    n_elements, # Total elements in matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    a = tl.load(A_ptr + offsets, mask=mask)
    
    # Perform multiplication
    out = a * s
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_scalar_mul(A: torch.Tensor, s: float):
    """
    Triton wrapper for matrix-scalar multiplication.
    """
    assert A.is_cuda, "Tensor must be on CUDA."
    
    # Ensure A is contiguous for linear indexing
    A = A.contiguous()
    M, N = A.shape
    n_elements = A.numel()
    
    # Prepare output tensor
    out = torch.empty_like(A)
    
    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    scalar_mul_kernel[grid](
        A, 
        out, 
        s, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication (C = A * s) using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using a custom Triton kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_scalar_mul(A, s)