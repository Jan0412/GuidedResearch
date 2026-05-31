import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    A_ptr,
    out_ptr,
    s,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the linear index for the start of the block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total number of elements
    num_elements = M * N
    
    # Create a mask to handle elements beyond the tensor size
    mask = offsets < num_elements
    
    # Load the input data
    # The tensor is contiguous, so we can access elements linearly
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    
    # Perform the multiplication
    out = A * s
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scale(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Wrapper function to launch the Triton kernel for matrix scaling.
    """
    assert A.is_cuda, "Input tensor must be on CUDA."
    A = A.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(A)
    
    M, N = A.shape
    num_elements = M * N
    
    # Tunable block size
    # 1024 is a good default for memory throughput
    BLOCK_SIZE = 1024
    
    # Define the grid
    # Number of blocks needed to cover all elements
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)
    
    # Launch the kernel
    scale_kernel[grid](A, out, s, M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for matrix-scalar multiplication.
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
        return triton_scale(A, s)