import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    
    # For each element in the block
    for i in range(BLOCK_SIZE):
        # Calculate global index
        idx = start_pos + i
        
        # Check bounds
        if idx >= n_elements:
            break
            
        # Calculate which row/sequence we're in
        row = idx // dim_size
        col = idx % dim_size
        
        # Initialize cumulative sum for this element
        cumsum = 0.0
        
        # Compute exclusive cumulative sum up to current position
        # We need to accumulate from position 0 to col-1
        for j in range(col):
            # Calculate the index in the input array
            input_idx = row * dim_size + j
            # Load value and accumulate
            val = tl.load(x_ptr + input_idx, mask=input_idx < n_elements, other=0.0)
            cumsum += val
            
        # Store the result
        output_idx = row * dim_size + col
        tl.store(output_ptr + output_idx, cumsum, mask=output_idx < n_elements)

@triton.jit
def fused_exclusive_cumsum_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    
    # Process elements in chunks to improve memory access patterns
    for chunk_start in range(start_pos, min(start_pos + BLOCK_SIZE, n_elements), 32):
        # For each element in this chunk
        for i in range(32):
            idx = chunk_start + i
            if idx >= n_elements:
                break
                
            # Calculate row and column
            row = idx // dim_size
            col = idx % dim_size
            
            # Compute exclusive cumulative sum
            cumsum = 0.0
            for j in range(col):
                input_idx = row * dim_size + j
                val = tl.load(x_ptr + input_idx, mask=input_idx < n_elements, other=0.0)
                cumsum += val
                
            # Store the result
            output_idx = row * dim_size + col
            tl.store(output_ptr + output_idx, cumsum, mask=output_idx < n_elements)

def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Compute exclusive cumulative sum using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    dim_size = shape[dim]
    n_elements = x.numel()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Handle special case where dim_size is 1
    if dim_size <= 1:
        return torch.zeros_like(x)
    
    # Calculate appropriate block size
    BLOCK_SIZE = 128
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Choose kernel based on problem size
    if dim_size > 1024:
        # Use the fused version for larger sequences
        fused_exclusive_cumsum_kernel[grid_size](
            x, output, n_elements, dim_size, 
            x.stride(dim), output.stride(dim),
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Use standard kernel for smaller sequences
        exclusive_cumsum_kernel[grid_size](
            x, output, n_elements, dim_size,
            x.stride(dim), output.stride(dim),
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return output

class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).
    Optimized with custom Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Directly use the Triton implementation
        return triton_exclusive_cumsum(x, self.dim)