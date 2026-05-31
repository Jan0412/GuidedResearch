import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reverse_cumsum_kernel(
    input_ptr,
    output_ptr,
    N,
    dim_size,
    stride_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID for the current thread
    block_id = tl.program_id(0)
    
    # Calculate the starting index for this block
    start_idx = block_id * BLOCK_SIZE
    
    # Handle the case where we're processing the last block
    if start_idx >= N:
        return
        
    # Calculate how many elements to process in this block
    num_elements = min(BLOCK_SIZE, N - start_idx)
    
    # Process elements in chunks of BLOCK_SIZE
    for i in range(num_elements):
        idx = start_idx + i
        # Calculate position in the flattened tensor
        pos = idx
        
        # Compute cumulative sum from right to left (reverse cumsum)
        # We'll do this by accumulating from the end of the dimension
        # to the beginning
        temp_sum = 0.0
        
        # For each element, we accumulate from the current position to the end
        # of the dimension in reverse order
        for j in range(dim_size - 1, -1, -1):
            # Calculate actual index in the tensor
            actual_idx = pos
            
            # Load value at this position
            val = tl.load(input_ptr + actual_idx, mask=(actual_idx < N))
            
            # Accumulate in reverse direction
            temp_sum += val
            
            # Store result at corresponding position in output
            tl.store(output_ptr + actual_idx, temp_sum, mask=(actual_idx < N))
            
            # Move to next element in the dimension (backwards)
            pos -= stride_dim

def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Custom Triton implementation of reverse cumulative sum
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    dim_size = shape[dim]
    
    # Calculate total elements and stride for the specified dimension
    N = x.numel()
    stride_dim = 1
    for i in range(dim + 1, len(shape)):
        stride_dim *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 128
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    reverse_cumsum_kernel[grid_size](
        x, out, N, dim_size, stride_dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for reverse cumulative sum
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Perform reverse cumulative sum using Triton kernel
        # This replaces the original: torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)
        return triton_reverse_cumsum(x, self.dim)