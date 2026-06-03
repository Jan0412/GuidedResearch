import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,           # Input tensor pointer
    out_ptr,         # Output tensor pointer (indices of minimum values)
    n_cols,          # Number of columns along the reduction dimension
    n_rows,          # Number of rows (other dimensions combined)
    stride_row,      # Stride between rows
    stride_col,      # Stride between columns in the reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (i.e., one output index)
    row_idx = tl.program_id(0)
    
    # Base pointer for this row
    x_row_ptr = x_ptr + row_idx * stride_row
    
    # Initialize minimum value and index
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    min_idx = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    
    # Process in blocks
    num_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        start = block_idx * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = start + offsets < n_cols
        
        # Load values
        col_offsets = start + offsets
        vals = tl.load(x_row_ptr + col_offsets * stride_col, mask=mask, other=float('inf'))
        
        # Compare and update minimums
        is_smaller = vals < min_val
        min_val = tl.where(is_smaller, vals, min_val)
        min_idx = tl.where(is_smaller, col_offsets, min_idx)
    
    # Reduce within the block to get final minimum index
    # Since we want minimal overhead and BLOCK_SIZE is manageable, we do simple reduction
    min_val = min_val[0]
    min_idx_final = min_idx[0]
    
    for i in range(1, BLOCK_SIZE):
        val_i = min_val
        idx_i = min_idx_final
        new_val = tl.where(val_i < min_val[i], val_i, min_val[i])
        new_idx = tl.where(val_i < min_val[i], idx_i, min_idx[i])
        min_val = new_val
        min_idx_final = new_idx
    
    # Store result
    out_ptr[row_idx] = min_idx_final


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of argmin along specified dimension.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to reduce along
        
    Returns:
        torch.Tensor: Indices of minimum values along dimension dim
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get shape information
    shape = x.shape
    dim_size = shape[dim]
    
    # Create output tensor
    out_shape = list(shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    
    # Reshape to 2D for easier processing: [rows, cols] where cols is dim_size
    # Move reduction dimension to last position
    if dim != -1 and dim != len(shape) - 1:
        x_perm = x.transpose(dim, -1).contiguous()
        out_perm = out.transpose(dim, -1).contiguous()
    else:
        x_perm = x
        out_perm = out
    
    # Reshape to 2D
    n_rows = x_perm.numel() // dim_size
    x_2d = x_perm.view(n_rows, dim_size)
    out_2d = out_perm.view(n_rows)
    
    # Calculate strides
    stride_row = x_2d.stride(0)
    stride_col = x_2d.stride(1)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(dim_size))
    
    # Launch grid
    grid = (n_rows,)
    
    # Launch kernel
    argmin_kernel[grid](
        x_2d,
        out_2d,
        dim_size,
        n_rows,
        stride_row,
        stride_col,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return to original shape
    if dim != -1 and dim != len(shape) - 1:
        return out_perm.transpose(dim, -1)
    else:
        return out


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension
    using a custom Triton kernel.
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