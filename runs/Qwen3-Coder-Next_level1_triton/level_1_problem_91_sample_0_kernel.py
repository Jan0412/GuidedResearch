import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    row_stride,
    col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate row pointer offset
    row_start = row_idx * row_stride
    
    # Initialize accumulator for reverse cumulative sum
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Process columns from right to left (reverse direction)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        # Calculate column indices for this block (in reverse order)
        col_indices = (n_cols - 1 - col_offset) - tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid columns
        mask = col_indices >= 0
        
        # Load values (in reverse order within the block)
        col_ptr = row_start + col_indices * col_stride
        vals = tl.load(col_ptr, mask=mask, other=0.0).to(tl.float32)
        
        # Compute reverse cumulative sum: acc = acc + vals
        acc = acc + vals
        
        # Store results (need to store in correct order)
        store_indices = col_indices
        store_ptr = row_start + store_indices * col_stride
        tl.store(store_ptr, acc.to(y_ptr.dtype.element_ty), mask=mask)


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Compute reverse cumulative sum along specified dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute reverse cumulative sum
        
    Returns:
        Tensor with reverse cumulative sum computed along dim
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and strides
    shape = x.shape
    stride = x.stride()
    
    # Convert negative dimension index
    if dim < 0:
        dim = len(shape) + dim
    
    # Calculate row and column dimensions for kernel
    # We'll treat the dimension as the "column" dimension for the kernel
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    
    n_cols = shape[dim]
    row_stride = stride[dim] if dim > 0 else 1
    for i in range(dim):
        if i > 0:
            row_stride *= shape[i]
    
    col_stride = stride[dim]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Grid: one program per row
    grid = (n_rows,)
    
    # Launch kernel
    reverse_cumsum_kernel[grid](
        x, out, n_rows, n_cols, row_stride, col_stride, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation along a specified dimension
    using custom Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)