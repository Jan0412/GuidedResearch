import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    A_ptr,
    out_ptr,
    s,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for matrix-scalar multiplication: out = A * s.
    """
    # Calculate the starting offset for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle elements beyond the tensor size
    mask = offsets < num_elements
    # Load data from A with masking
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    # Perform element-wise multiplication with scalar s
    out = A * s
    # Store the result back to out with masking
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Wrapper function to launch the Triton scale kernel.
    """
    assert A.is_cuda, "Input tensor A must be on CUDA."
    A = A.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(A)
    
    # Total number of elements
    num_elements = A.numel()
    
    # Tunable block size; 1024 is a good default for memory-bound operations
    BLOCK_SIZE = 1024
    
    # Grid calculation: number of blocks needed
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    scale_kernel[grid](A, out, s, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix-scalar multiplication.
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