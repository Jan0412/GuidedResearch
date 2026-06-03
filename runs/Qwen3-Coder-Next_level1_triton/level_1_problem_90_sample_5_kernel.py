import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    n_rows,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Determine if we're cumprod along rows (dim=1) or columns (dim=0)
    if dim == 1:
        # Process rows: each program handles one row
        row_id = tl.program_id(0)
        row_start = row_id * n_cols
        
        # Initialize the running product
        running_prod = tl.load(x_ptr + row_start)  # First element
        tl.store(out_ptr + row_start, running_prod)
        
        # Process remaining elements in the row
        for col_offset in range(1, n_cols):
            col_idx = row_start + col_offset
            x_val = tl.load(x_ptr + col_idx)
            running_prod = running_prod * x_val
            tl.store(out_ptr + col_idx, running_prod)
    else:  # dim == 0
        # Process columns: each program handles one column
        col_id = tl.program_id(0)
        col_start = col_id
        
        # Initialize the running product
        running_prod = tl.load(x_ptr + col_start)  # First element
        tl.store(out_ptr + col_start, running_prod)
        
        # Process remaining elements in the column
        for row_offset in range(1, n_rows):
            row_idx = col_start + row_offset * n_cols
            x_val = tl.load(x_ptr + row_idx)
            running_prod = running_prod * x_val
            tl.store(out_ptr + row_idx, running_prod)


def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative product along a specified dimension.
    
    Args:
        x (torch.Tensor): Input tensor (FP32)
        dim (int): Dimension along which to compute cumulative product
        
    Returns:
        torch.Tensor: Output tensor with same shape as input
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    
    # Get tensor dimensions
    shape = x.shape
    n_dims = len(shape)
    
    # Adjust negative dimension index
    if dim < 0:
        dim = n_dims + dim
        
    # For 2D case (which is what our input is), handle both dimensions
    if n_dims == 2:
        n_rows, n_cols = shape[0], shape[1]
        
        if dim == 1:
            # Process along columns (each row is a sequence)
            grid = (n_rows,)
            BLOCK_SIZE = 128  # Not used for row-wise but kept for interface
            cumprod_kernel[grid](x, out, n_cols, n_rows, dim=dim, BLOCK_SIZE=BLOCK_SIZE)
        elif dim == 0:
            # Process along rows (each column is a sequence)
            grid = (n_cols,)
            BLOCK_SIZE = 128
            cumprod_kernel[grid](x, out, n_cols, n_rows, dim=dim, BLOCK_SIZE=BLOCK_SIZE)
        else:
            raise ValueError(f"Dimension {dim} out of range for 2D tensor")
    else:
        # For higher dimensions, use pytorch fallback (less common case)
        return torch.cumprod(x, dim=dim)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of CumulativeProductModel using Triton kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, dim=self.dim)