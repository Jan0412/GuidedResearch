import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch dimension)
    n_cols,  # Number of columns (sequence dimension)
    stride_row,  # Stride for row dimension
    stride_col,  # Stride for column dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate starting offsets for this row
    row_start = row_idx * stride_row
    
    # Initialize accumulator for exclusive sum (starts with 0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        # Create column offsets
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid columns
        mask = col_offsets < n_cols
        
        # Load input values
        x_offsets = row_start + col_offsets * stride_col
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Store current accumulator (this is the exclusive sum)
        tl.store(out_ptr + x_offsets, acc.to(out_ptr.dtype.element_ty), mask=mask)
        
        # Update accumulator with current values
        acc = acc + x


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Compute exclusive cumulative sum along specified dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute exclusive cumulative sum
    
    Returns:
        Tensor with exclusive cumulative sum
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and dimensions
    shape = x.shape
    ndim = x.ndim
    
    # Normalize negative dimension index
    if dim < 0:
        dim = ndim + dim
    
    # Get sizes
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    n_cols = shape[dim]
    row_stride = x.stride(dim)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine block size
    BLOCK_SIZE = 256
    
    # Compute grid dimensions
    grid = (n_rows,)
    
    # Launch kernel
    exclusive_cumsum_kernel[grid](
        x, out,
        n_rows, n_cols,
        x.stride()[:dim] + x.stride()[dim+1:] if dim < ndim - 1 else x.stride()[:-1],
        row_stride,
        BLOCK_SIZE=BLOCK_SIZE
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
        # Use the optimized Triton exclusive cumsum kernel
        return triton_exclusive_cumsum(x, self.dim)