import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def exclusive_cumsum_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting offset for this block
    start = block_idx * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < N
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute exclusive cumulative sum
    # For exclusive cumsum, we want each element to contain sum of all previous elements
    # We'll use a simple approach: load, scan, store
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Forward pass: compute inclusive prefix sum
    for i in range(BLOCK_SIZE):
        if i == 0:
            accumulator[i] = input_data[i]
        else:
            accumulator[i] = accumulator[i-1] + input_data[i]
    
    # Shift the result to make it exclusive
    # The first element should be 0, the rest should be shifted left
    exclusive_result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    exclusive_result[0] = 0.0
    for i in range(1, BLOCK_SIZE):
        exclusive_result[i] = accumulator[i-1]
    
    # Store the result
    tl.store(output_ptr + offsets, exclusive_result, mask=mask)

def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Triton implementation of exclusive cumulative sum along a specified dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    assert dim == 1, "Currently only supports dim=1"
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get dimensions
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    
    # Flatten to 1D for processing
    flat_x = x.view(-1)
    flat_output = torch.empty_like(flat_x)
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid_size = (flat_x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    exclusive_cumsum_kernel[grid_size](
        flat_x,
        flat_output,
        flat_x.numel(),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to original shape
    return flat_output.view(x.shape)

class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).
    Optimized version using Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use our Triton kernel instead of the PyTorch operations
        return triton_exclusive_cumsum(x, self.dim)