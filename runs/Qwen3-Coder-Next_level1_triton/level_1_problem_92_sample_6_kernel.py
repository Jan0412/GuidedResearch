import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    stride,  # Stride along the dimension we're cumsumming
    dim_size,  # Size of the dimension we're cumsumming
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one "row" (one set of elements along the cumsum dimension)
    row_idx = tl.program_id(0)
    
    # Calculate base offset for this row
    base_offset = row_idx * stride
    
    # We'll do a parallel prefix sum using a scan algorithm
    # Load the current row elements
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize prefix sum
    prefix_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # For the first element, exclusive cumsum is always 0
    if offsets[0] == 0:
        tl.store(out_ptr + base_offset, 0.0)
    
    # Process elements
    for i in range(dim_size):
        # Load current element
        current_offset = base_offset + i * stride
        mask = (current_offset < n_elements) & (offsets == 0)
        if i < dim_size:
            elem = tl.load(x_ptr + current_offset, mask=mask, other=0.0)
            
            # Store the current prefix sum as the exclusive cumsum for position i+1
            if i + 1 < dim_size:
                out_offset = base_offset + (i + 1) * stride
                tl.store(out_ptr + out_offset, prefix_sum[0], mask=(out_offset < n_elements))
            
            # Update prefix sum for next iteration
            if i < dim_size - 1:
                prefix_sum = prefix_sum + elem


@triton.jit
def exclusive_cumsum_kernel_v2(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    dim_size,  # Size of the dimension we're cumsumming
    stride,  # Stride along the dimension we're cumsumming
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one "row" (one set of elements along the cumsum dimension)
    row_idx = tl.program_id(0)
    
    # Calculate base offset for this row
    base_offset = row_idx * stride
    
    # Process elements sequentially
    prefix_sum = 0.0
    for i in range(dim_size):
        current_offset = base_offset + i * stride
        
        # Store the current prefix sum as the exclusive cumsum for position i
        tl.store(out_ptr + current_offset, prefix_sum)
        
        # Load current element and update prefix sum
        elem = tl.load(x_ptr + current_offset)
        prefix_sum = prefix_sum + elem


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Compute exclusive cumulative sum along a given dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute exclusive cumsum
        
    Returns:
        Tensor with exclusive cumulative sum
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate strides and dimensions
    shape = x.shape
    dim_size = shape[dim]
    
    # Calculate stride along the cumsum dimension
    stride = 1
    for i in range(dim + 1, len(shape)):
        stride *= shape[i]
    
    # Calculate number of rows (batches)
    num_rows = 1
    for i in range(dim):
        num_rows *= shape[i]
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Launch kernel
    grid = (num_rows,)
    exclusive_cumsum_kernel_v2[grid](
        x, out, dim_size, stride, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum using Triton kernels.

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use our Triton-based exclusive cumsum
        return triton_exclusive_cumsum(x, self.dim)