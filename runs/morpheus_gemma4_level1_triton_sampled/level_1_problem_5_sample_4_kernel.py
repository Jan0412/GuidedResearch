import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scalar_mul_kernel(
    A_ptr,      # Pointer to input matrix
    C_ptr,      # Pointer to output matrix
    s,          # Scalar value
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID and calculate the range of offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load elements from input matrix A
    a = tl.load(A_ptr + offsets, mask=mask)
    
    # Perform scalar multiplication
    c = a * s
    
    # Store result in output matrix C
    tl.store(C_ptr + offsets, c, mask=mask)

def triton_scalar_mul(A: torch.Tensor, s: float):
    """
    Triton wrapper for matrix-scalar multiplication.
    """
    # Ensure tensor is on CUDA and contiguous
    assert A.is_cuda, "Tensor must be on CUDA"
    A = A.contiguous()
    
    # Output tensor
    C = torch.empty_like(A)
    
    n_elements = A.numel()
    # Block size for the kernel
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    scalar_mul_kernel[grid](
        A, C, s, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication using a Triton kernel.
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