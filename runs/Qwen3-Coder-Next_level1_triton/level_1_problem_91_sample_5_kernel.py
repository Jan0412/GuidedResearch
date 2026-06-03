import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    n_rows,
    stride_row,
    stride_col,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate pointer offset for this row
    row_start = row_idx * stride_row
    
    # Initialize accumulator for reverse cumulative sum
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Process from right to left (reverse direction)
    for col_idx in range(n_cols - 1, -1, -1):
        # Calculate offset for current position
        offset = row_start + col_idx * stride_col
        
        # Load value
        val = tl.load(x_ptr + offset).to(tl.float32)
        
        # Add to accumulator
        acc = acc + val
        
        # Store accumulated value
        tl.store(out_ptr + offset, acc.to(x_ptr.dtype.element_ty))


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Compute reverse cumulative sum along specified dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute reverse cumulative sum
        
    Returns:
        Tensor with reverse cumulative sum computed
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and dimensions
    shape = x.shape
    ndim = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim += ndim
    
    # Calculate strides
    stride = x.stride()
    
    # Calculate dimensions for the kernel
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    
    n_cols = shape[dim]
    stride_row = stride[dim] if dim == 0 else 1
    stride_col = 1
    
    # Calculate actual row stride for the kernel
    for i in range(dim):
        stride_row *= shape[i] if i > 0 else 1
    
    # Adjust strides for proper indexing
    stride_row = 1
    for i in range(dim):
        stride_row *= shape[i]
    
    stride_col = stride[dim]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = min(128, n_cols)
    
    # Determine grid dimensions
    grid = (n_rows,)
    
    # Launch the Triton kernel
    reverse_cumsum_kernel[grid](
        x,
        out,
        n_cols,
        n_rows,
        stride_row,
        stride_col,
        BLOCK_SIZE=BLOCK_SIZE
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
        # Use our optimized Triton kernel for reverse cumulative sum
        return triton_reverse_cumsum(x, self.dim)