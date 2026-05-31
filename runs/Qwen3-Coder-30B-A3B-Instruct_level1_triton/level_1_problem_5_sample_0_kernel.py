import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mul_scalar_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    scalar,   # Scalar value
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise multiplication with scalar
    out = x * scalar
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_mul_scalar(x: torch.Tensor, scalar: float):
    """
    This function wraps the Triton kernel call for matrix-scalar multiplication.
    It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    mul_scalar_kernel[grid](x, out, scalar, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication using Triton kernel
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
        if A.is_cuda:
            return triton_mul_scalar(A, s)
        else:
            return A * s