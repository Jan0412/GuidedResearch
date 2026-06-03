import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows
    n_cols,  # Number of columns
    stride_row,  # Stride between rows
    stride_col,  # Stride between columns
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Compute starting pointer for this row
    row_start = x_ptr + row_idx * stride_row
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator for exclusive cumsum
    # For exclusive cumsum, we start with 0 and accumulate previous elements
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process the row
    for col_idx in range(0, n_cols, BLOCK_SIZE):
        # Compute actual column offsets
        actual_cols = col_offsets + col_idx
        mask = actual_cols < n_cols
        
        # Load current element
        x_val = tl.load(row_start + actual_cols * stride_col, mask=mask, other=0.0).to(tl.float32)
        
        # Store the current accumulator (exclusive - doesn't include current element)
        tl.store(out_ptr + row_idx * stride_row + actual_cols * stride_col, acc, mask=mask)
        
        # Update accumulator with current element for next iteration
        acc = acc + x_val


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Compute exclusive cumulative sum along specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute exclusive cumsum
        
    Returns:
        Tensor with exclusive cumulative sum
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    dim_size = shape[dim]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate strides
    stride = x.stride()
    dim_stride = stride[dim]
    
    # Reshape to 2D for easier processing: (n_rows, n_cols)
    # where n_cols is the dimension size along dim
    if dim == 0:
        # Special case: dim=0, we want to process along dim
        n_rows = 1
        n_cols = dim_size
        stride_row = 1
        stride_col = dim_stride
    else:
        # Reshape: (product of dims before dim, product of dims after dim)
        n_rows = 1
        for i in range(dim):
            n_rows *= shape[i]
        n_cols = dim_size
        stride_row = dim_stride
        stride_col = 1 if dim == len(shape) - 1 else stride[dim + 1]
    
    # For simplicity, we'll handle the general case by permuting if needed
    # Actually, let's do a simpler approach - process along last dimension
    if dim != len(shape) - 1:
        # Permute to move dim to last position
        dims = list(range(len(shape)))
        dims.append(dims.pop(dim))
        x_permuted = x.permute(dims)
        out_permuted = triton_exclusive_cumsum(x_permuted, len(shape) - 1)
        # Permute back
        restore_dims = list(range(len(shape)))
        restore_dims.insert(dim, restore_dims.pop())
        return out_permuted.permute(restore_dims)
    
    # Now dim is the last dimension, process each row
    BLOCK_SIZE = 128
    grid = lambda meta: (n_rows,)
    
    exclusive_cumsum_kernel[grid](
        x, out, n_rows, n_cols, 
        x.stride(0) if len(shape) > 1 else 1, 
        dim_stride,
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
        return triton_exclusive_cumsum(x, self.dim)