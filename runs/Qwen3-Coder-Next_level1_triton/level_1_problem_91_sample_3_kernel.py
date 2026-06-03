import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (flattened dimension except the target dim)
    n_cols,  # Size of the dimension along which to compute reverse cumsum
    row_stride,  # Stride between rows in memory
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Compute base pointers for this row
    row_start = row_idx * row_stride
    x_row_ptr = x_ptr + row_start
    out_row_ptr = out_ptr + row_start
    
    # Initialize running sum to zero
    cumsum = 0.0
    
    # Traverse the row in reverse order
    # We'll use a loop that goes from the end to the beginning
    for col_offset in range(n_cols - 1, -1, -1):
        # Compute pointer to current element
        col_ptr = row_start + col_offset
        
        # Load current element
        x_val = tl.load(x_row_ptr + col_offset)
        
        # Update cumulative sum
        cumsum = cumsum + x_val
        
        # Store result
        tl.store(out_row_ptr + col_offset, cumsum)


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Compute reverse cumulative sum along a specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute reverse cumulative sum
        
    Returns:
        Tensor with reverse cumulative sum computed along dim
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get shape and ensure dim is valid
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    assert dim < len(shape), f"Dimension {dim} out of range for shape {shape}"
    
    # Compute the number of rows and size of the dimension to process
    n_rows = 1
    for i, s in enumerate(shape):
        if i != dim:
            n_rows *= s
    n_cols = shape[dim]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Determine row stride (number of elements between consecutive rows)
    row_stride = 1
    for i in range(dim + 1, len(shape)):
        row_stride *= shape[i]
    
    # Set block size - tuned for good performance
    BLOCK_SIZE = 128
    
    # Launch grid: one program per row
    grid = (n_rows,)
    
    # Launch the kernel
    reverse_cumsum_kernel[grid](
        x, out, n_rows, n_cols, row_stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs reverse cumulative sum using Triton kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)