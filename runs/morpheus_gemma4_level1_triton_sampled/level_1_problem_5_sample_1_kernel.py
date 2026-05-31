import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scalar_mul_kernel(
    A_ptr,      # Pointer to input matrix A
    s,          # Scalar value
    out_ptr,    # Pointer to output matrix
    n_elements, # Total number of elements (M * N)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    a = tl.load(A_ptr + offsets, mask=mask)
    
    # Perform scalar multiplication
    out = a * s
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_scalar_mul(A: torch.Tensor, s: float):
    """
    Triton wrapper for matrix-scalar multiplication.
    """
    assert A.is_cuda, "Tensor must be on CUDA."
    
    # Ensure A is contiguous for efficient pointer arithmetic
    A = A.contiguous()
    M, N = A.shape
    n_elements = M * N
    
    # Prepare output tensor
    out = torch.empty_like(A)
    
    # BLOCK_SIZE is a tunable parameter. 1024 is generally a good starting point for element-wise ops.
    BLOCK_SIZE = 1024
    
    # Calculate the number of blocks needed to cover all elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    scalar_mul_kernel[grid](
        A, 
        s, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_scalar_mul(A, s)