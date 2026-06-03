import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer (indices)
    n_rows,  # Number of rows (all dimensions except the argmin dimension)
    n_cols,  # Size of the argmin dimension
    row_stride,  # Stride to move to next row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one output index)
    row_idx = tl.program_id(0)
    
    # Calculate pointer to the start of this row
    x_row_ptr = x_ptr + row_idx * row_stride
    out_row_ptr = out_ptr + row_idx
    
    # Initialize minimum value and index
    min_val = tl.full([1], float('inf'), dtype=tl.float32)
    min_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min = tl.min(x)
        block_idx = tl.argmin(x, axis=0)
        
        # Update global minimum if needed
        is_smaller = block_min < min_val
        min_val = tl.where(is_smaller, block_min, min_val)
        min_idx = tl.where(is_smaller, block_idx + start_col, min_idx)
    
    # Store result
    tl.store(out_row_ptr, min_idx)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Custom Triton implementation of argmin.
    
    Args:
        x: Input tensor
        dim: Dimension along which to find minimum indices
        
    Returns:
        Tensor containing indices of minimum values along specified dimension
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape
    shape = x.shape
    n_dims = len(shape)
    
    # Normalize negative dimension index
    if dim < 0:
        dim = n_dims + dim
    
    # Calculate dimensions for the kernel
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    
    n_cols = shape[dim]
    
    # Calculate row stride (product of dimensions after dim)
    row_stride = 1
    for i in range(dim + 1, n_dims):
        row_stride *= shape[i]
    
    # Create output tensor
    out_shape = list(shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=torch.long, device=x.device)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (n_rows,)
    
    # Launch kernel
    argmin_kernel[grid](x, out, n_rows, n_cols, row_stride, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension
    using custom Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return triton_argmin(x, self.dim)