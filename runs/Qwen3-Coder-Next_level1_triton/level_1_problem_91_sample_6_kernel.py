import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    dim_size,  # Size of the dimension we're summing over
    other_dim_size,  # Size of the other dimensions (product of all dims except the target dim)
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the global block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Initialize output with zeros
    out = tl.zeros_like(x)
    
    # For each "row" in the dimension we're summing over, compute reverse cumulative sum
    # We process in chunks to handle the dimension properly
    
    # We need to compute cumulative sum from end to start along the specified dimension
    # For each position i in the dimension, out[i] = sum(x[i], x[i+1], ..., x[dim_size-1])
    
    # Process in reverse order
    for i in range(dim_size - 1, -1, -1):
        # Compute offset for current position in the dimension
        # We'll process elements that correspond to position i in the dimension
        
        # Create mask for elements at position i in the dimension
        # For a 2D case, if dim=1, then for row r and column c, the index is r * dim_size + c
        # For higher dimensions, we need to compute the position in the dimension
        
        # Calculate the position within the dimension for each offset
        # This requires some index arithmetic based on the dimension
        
        # For simplicity, we'll process each "slice" separately
        pass  # We'll implement a more direct approach below


@triton.jit
def reverse_cumsum_optimized_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    dim_size,  # Size of the dimension we're summing over
    other_dim_size,  # Size of the other dimensions (product of all dims except the target dim)
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # For each "row" along the specified dimension, compute reverse cumulative sum
    # We iterate over each row and compute the reverse cumulative sum
    
    # Process each row in parallel
    row_id = tl.program_id(0)
    
    # Start index for this row
    row_start = row_id * dim_size
    
    # Create offsets for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_start + dim_size)
    
    # Load the row data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute reverse cumulative sum
    # Start from the end and work backwards
    cumsum = tl.zeros_like(x)
    
    # Process from end to beginning
    for i in range(dim_size - 1, -1, -1):
        # For the last element
        if i == dim_size - 1:
            cumsum = tl.where(offsets == (row_start + i), x, cumsum)
        else:
            # Add current element to the cumulative sum from the right
            next_cumsum = tl.load(out_ptr + (row_start + i + 1)) if (row_start + i + 1) < n_elements else 0.0
            cumsum = tl.where(offsets == (row_start + i), x + next_cumsum, cumsum)
    
    # Store the result
    tl.store(out_ptr + offsets, cumsum, mask=mask)


@triton.jit
def reverse_cumsum_fused_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    dim_size,  # Size of the dimension we're summing over
    other_dim_size,  # Size of the other dimensions (product of all dims except the target dim)
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # More efficient implementation that processes rows sequentially within each block
    # but processes each row from end to beginning in one pass
    
    # Process multiple rows per block for better occupancy
    row_stride = tl.num_programs(1) if tl.num_programs(0) > 1 else 1
    num_rows_per_block = tl.num_programs(0) if tl.num_programs(1) == 1 else 1
    
    for row_offset in range(num_rows_per_block):
        row_id = tl.program_id(1) + row_offset * tl.num_programs(1)
        if row_id >= other_dim_size:
            break
            
        # Start index for this row
        row_start = row_id * dim_size
        
        # Initialize cumulative sum to zero
        cumsum = tl.zeros(1, dtype=tl.float32)
        
        # Process from end to beginning
        for i in range(dim_size - 1, -1, -1):
            # Compute offset for current position
            idx = row_start + i
            mask = idx < n_elements
            
            # Load current element
            x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
            
            # Update cumulative sum
            cumsum = cumsum + x_val
            
            # Store result
            tl.store(out_ptr + idx, cumsum, mask=mask)


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Compute reverse cumulative sum along specified dimension using Triton kernel.
    
    reverse_cumsum(x, dim) = flip(cumsum(flip(x, dim), dim), dim)
    
    But computed directly without flips for better performance.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    n_elements = x.numel()
    dim_size = shape[dim]
    other_dim_size = n_elements // dim_size
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine grid dimensions
    # We want to parallelize over the "rows" perpendicular to the target dimension
    # Each row has length dim_size
    
    # For simplicity, we'll use 1D grid where each block processes one row
    # But to handle large dim_size, we might need multiple blocks per row
    
    BLOCK_SIZE = 128
    
    # Grid: process all rows in parallel
    grid = (other_dim_size,)
    
    # Launch kernel
    reverse_cumsum_fused_kernel[grid](
        x, out, n_elements, dim_size, other_dim_size, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs reverse cumulative sum using Triton kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)