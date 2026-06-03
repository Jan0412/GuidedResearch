import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mul_kernel(
    a_ptr,  # Pointer to input matrix A
    s,  # Scalar value
    out_ptr,  # Pointer to output matrix
    n_elements,  # Total number of elements in the matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise multiplication with scalar
    out = a * s
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scalar_mul(a: torch.Tensor, s: float) -> torch.Tensor:
    """
    This function wraps the Triton kernel call for matrix-scalar multiplication. It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert a.is_cuda, "Tensor must be on CUDA."
    a = a.contiguous()

    # Prepare output tensor
    out = torch.empty_like(a)

    # Number of elements in the tensor
    n_elements = a.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    scalar_mul_kernel[grid](a, s, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication (C = A * s)
    using a custom Triton kernel instead of PyTorch's built-in operation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using a Triton kernel.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_scalar_mul(A, s)