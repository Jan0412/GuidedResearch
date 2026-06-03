import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride,
    col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index this program instance handles
    row_id = tl.program_id(0)
    
    # Calculate starting pointer for this row
    row_start = x_ptr + row_id * row_stride
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator for reverse cumulative sum
    # We'll accumulate from the end of the row backward
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process columns in reverse order
    # We need to handle rows that might be longer than BLOCK_SIZE
    for col_base in range(n_cols - 1, -1, -BLOCK_SIZE):
        # Calculate actual column offsets for this iteration
        current_offsets = col_offsets + col_base
        mask = current_offsets < n_cols
        
        # Load data
        x = tl.load(row_start + current_offsets * col_stride, mask=mask, other=0.0)
        
        # Convert to float32 for accumulation
        x_f32 = x.to(tl.float32)
        
        # Add to accumulator (reverse cumulative sum)
        acc = acc + x_f32
        
        # Store result (in reverse order, we'll flip later or store directly)
        tl.store(out_ptr + row_id * row_stride + current_offsets * col_stride, acc.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def reverse_cumsum_fused_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride,
    col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index this program instance handles
    row_id = tl.program_id(0)
    
    # Calculate starting pointer for this row
    row_start = x_ptr + row_id * row_stride
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator for reverse cumulative sum
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process columns in reverse order
    for col_base in range(n_cols - 1, -1, -BLOCK_SIZE):
        # Calculate actual column offsets for this iteration
        current_offsets = col_offsets + col_base
        mask = current_offsets < n_cols
        
        # Load data
        x = tl.load(row_start + current_offsets * col_stride, mask=mask, other=0.0)
        
        # Convert to float32 for accumulation
        x_f32 = x.to(tl.float32)
        
        # Add to accumulator (reverse cumulative sum)
        acc = acc + x_f32
        
        # Store result directly to output
        tl.store(out_ptr + row_id * row_stride + current_offsets * col_stride, acc.to(x_ptr.dtype.element_ty), mask=mask)


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Compute reverse cumulative sum along specified dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute reverse cumulative sum
        
    Returns:
        Result tensor with same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    n_rows = 1
    n_cols = shape[dim]
    
    # Calculate row stride and column stride
    strides = x.stride()
    row_stride = strides[0] if dim == 0 else sum(strides[:dim])
    col_stride = strides[dim]
    
    # For 1D case, handle specially
    if len(shape) == 1:
        n_rows = shape[0]
        n_cols = 1
        row_stride = 1
        col_stride = 0
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine block size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    reverse_cumsum_fused_kernel[grid](
        x,
        out,
        n_rows,
        n_cols,
        row_stride,
        col_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation along a specified dimension using Triton kernels.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use the optimized Triton implementation
        return triton_reverse_cumsum(x, self.dim)