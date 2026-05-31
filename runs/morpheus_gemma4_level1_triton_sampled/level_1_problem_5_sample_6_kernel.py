import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mul_scalar_kernel(
    x_ptr,      # Pointer to input matrix
    out_ptr,    # Pointer to output matrix
    s,          # Scalar value
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform scalar multiplication
    out = x * s
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_mul_scalar(x: torch.Tensor, s: float):
    """
    Wraps the Triton kernel for matrix-scalar multiplication.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    
    # Ensure input is contiguous
    x = x.contiguous()
    n_elements = x.numel()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Block size for the kernel
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    # Launch kernel
    mul_scalar_kernel[grid](
        x, 
        out, 
        s, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication using Triton.
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
        return triton_mul_scalar(A, s)