import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reverse_cumsum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Handle the case where we're processing elements in the dimension
    # We need to process each element in the batch
    if stride == 1:
        # For the last dimension, we can use simpler indexing
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        # Compute reverse cumulative sum
        cumsum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in range(dim_size - 1, -1, -1):
            # Calculate offset for this element
            current_offset = i * stride
            if current_offset < n_elements:
                # Load current value
                val = tl.load(input_ptr + current_offset, mask=(current_offset < n_elements), other=0.0)
                # Accumulate from right to left
                cumsum = tl.where(i == dim_size - 1, val, cumsum + val)
                # Store result
                tl.store(output_ptr + current_offset, cumsum, mask=(current_offset < n_elements))
    else:
        # General case for other dimensions
        # We'll compute it differently to handle strides properly
        mask = offsets < n_elements
        # Load all values for this block
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Compute cumulative sum from right to left manually for small blocks
        # This is a simplified approach that works when BLOCK_SIZE <= dim_size
        for i in range(dim_size - 1, -1, -1):
            current_offset = i * stride
            if current_offset < n_elements:
                # Accumulate from right to left
                cumsum_val = tl.zeros((1,), dtype=tl.float32)
                for j in range(i, dim_size):
                    j_offset = j * stride
                    if j_offset < n_elements:
                        val = tl.load(input_ptr + j_offset, mask=(j_offset < n_elements), other=0.0)
                        cumsum_val += val
                tl.store(output_ptr + current_offset, cumsum_val, mask=(current_offset < n_elements))

# More efficient approach using proper block-wise processing
@triton.jit
def reverse_cumsum_kernel_v2(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Process along the specific dimension
    block_idx = tl.program_id(0)
    
    # For each block, we process one slice along the dimension
    # but we still need to do the full computation
    
    # Simplified approach: 
    # Each thread processes one element of the output array
    # But we need to do the reverse cumulative sum correctly
    
    # Let's restructure this for better performance
    # We'll compute cumulative sums row by row
    row_start = block_idx * dim_size
    
    # Process the reverse cumulative sum for the entire row
    # First load all elements of this row
    row_offsets = tl.arange(0, dim_size) * stride
    mask = row_offsets < n_elements
    values = tl.load(input_ptr + row_offsets, mask=mask, other=0.0)
    
    # Compute reverse cumulative sum
    cumsum = tl.zeros((dim_size,), dtype=tl.float32)
    for i in range(dim_size - 1, -1, -1):
        cumsum[i] = values[i] + (cumsum[i+1] if i+1 < dim_size else 0.0)
    
    # Store results
    tl.store(output_ptr + row_offsets, cumsum, mask=mask)

# Even better approach: Process elements in a way that matches the pattern
@triton.jit
def reverse_cumsum_kernel_optimized(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block handles one row along the specified dimension
    block_idx = tl.program_id(0)
    
    # Each block processes one row (along the dimension we care about)
    # We assume that we have at least dim_size elements per row
    row_start = block_idx * dim_size
    
    # Load all elements in this row
    offsets = row_start + tl.arange(0, dim_size)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute reverse cumulative sum
    # Start from the end and work backwards
    cumsum = tl.zeros((dim_size,), dtype=tl.float32)
    for i in range(dim_size - 1, -1, -1):
        # This approach uses the fact that we process in reverse order
        # and accumulate from right to left
        if i == dim_size - 1:
            cumsum[i] = x[i]
        else:
            cumsum[i] = x[i] + cumsum[i + 1]
    
    # Store results
    tl.store(output_ptr + offsets, cumsum, mask=mask)

def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Computes reverse cumulative sum along specified dimension using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get the shape and calculate total elements
    shape = x.shape
    dim_size = shape[dim]
    n_elements = x.numel()
    
    # Calculate stride for the specified dimension
    stride = 1
    for i in range(dim + 1, len(shape)):
        stride *= shape[i]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # If we have more than 4096 elements, we might need to adjust block size
    BLOCK_SIZE = 1024
    if dim_size > 1024:
        BLOCK_SIZE = 512
    
    # Grid size - one block per row along the dimension
    grid_size = (n_elements + dim_size - 1) // dim_size
    
    # Launch kernel
    reverse_cumsum_kernel_optimized[grid_size](
        x, out, n_elements, dim_size, stride, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.
    Optimized with custom Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Since we're doing flip(cumsum(flip(x))), we can optimize this
        # by implementing a direct reverse cumulative sum kernel
        
        # Use our Triton implementation
        return triton_reverse_cumsum(x, self.dim)