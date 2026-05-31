import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduction_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one reduction)
    row_idx = tl.program_id(0)
    
    # Calculate row offset
    row_start = row_idx * n_cols
    
    # Initialize accumulator
    sum_val = tl.zeros([1], dtype=tl.float32)
    
    # Process elements in chunks
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Accumulate
        sum_val += tl.sum(x, axis=0)
    
    # Store result
    tl.store(out_ptr + row_idx, sum_val)


def triton_sum_reduction(x: torch.Tensor, dim: int):
    """
    Performs sum reduction along the specified dimension using Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Make tensor contiguous for efficient memory access
    x = x.contiguous()
    
    # Get shape and compute dimensions
    shape = x.shape
    n_dims = len(shape)
    
    # Normalize negative dimension index
    if dim < 0:
        dim += n_dims
    
    # Compute number of rows and columns for the reduction
    # We'll flatten all dimensions before 'dim' into rows
    # and all dimensions after 'dim' into columns for simplicity
    if dim == 0:
        # Reduce first dimension: treat as (1, prod(dimensions)) reduction
        n_rows = 1
        n_cols = x.numel()
    elif dim == n_dims - 1:
        # Reduce last dimension: treat as (prod(previous_dims), last_dim) reduction
        n_rows = x.numel() // shape[-1]
        n_cols = shape[-1]
    else:
        # Reduce middle dimension
        # Calculate size before and after the reduction dimension
        n_before = 1
        for i in range(dim):
            n_before *= shape[i]
        
        n_after = 1
        for i in range(dim + 1, n_dims):
            n_after *= shape[i]
        
        n_rows = n_before * n_after
        n_cols = shape[dim]
    
    # Prepare output tensor
    out_shape = list(shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Determine block size
    BLOCK_SIZE = min(1024, n_cols)
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    sum_reduction_kernel[grid](x, out.reshape(-1), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return triton_sum_reduction(x, self.dim)