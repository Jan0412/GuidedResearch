import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor (indices)
    n_rows,  # Number of rows (batch dimension)
    n_cols,  # Number of columns (size of dimension along which to argmin)
    stride_row,  # Stride between rows
    stride_col,  # Stride between columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Pointer to the start of this row
    x_row_ptr = x_ptr + row_idx * stride_row
    
    # Initialize min value and index
    min_val = tl.float32(1e10)  # Large initial value for FP32
    min_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    
    # Process columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        # Compute actual column offsets
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        # Mask to ensure we don't go out of bounds
        mask = offsets < n_cols
        
        # Load current values
        curr_vals = tl.load(x_row_ptr + offsets * stride_col, mask=mask, other=1e10)
        
        # For the first block, set indices to actual positions
        if start_col == 0:
            curr_indices = offsets
        else:
            curr_indices = start_col + tl.arange(0, BLOCK_SIZE)
        
        # Update min value and index where appropriate
        is_smaller = curr_vals < min_val
        min_val = tl.where(is_smaller, curr_vals, min_val)
        min_idx = tl.where(is_smaller, curr_indices, min_idx)
    
    # Now find the minimum among the candidates in min_idx
    # We'll do a sequential reduction since BLOCK_SIZE is typically small (e.g., 32 or 64)
    best_val = min_val[0]
    best_idx = min_idx[0]
    for i in range(1, BLOCK_SIZE):
        curr_val = min_val[i]
        curr_idx = min_idx[i]
        is_smaller = curr_val < best_val
        best_val = tl.where(is_smaller, curr_val, best_val)
        best_idx = tl.where(is_smaller, curr_idx, best_idx)
    
    # Store the result
    tl.store(out_ptr + row_idx, best_idx)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes argmin along specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute argmin
        
    Returns:
        Tensor with indices of minimum values along the specified dimension
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions and strides
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    assert 0 <= dim < len(shape), "Invalid dimension"
    
    # Determine row and column dimensions
    # We treat everything before dim as rows and everything after as part of columns
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    
    n_cols = shape[dim]
    stride_row = x.stride(dim) if dim == len(shape) - 1 else x.stride(dim) * n_cols
    stride_col = 1
    
    # Calculate strides properly
    # For a tensor with shape (d0, d1, ..., dn), the stride for dimension i is:
    # stride[i] = product of sizes of dimensions i+1 to n
    # So for dimension 'dim', stride is product of dimensions after dim
    stride_dim = x.stride(dim)
    
    # Reshape tensor if needed to 2D for easier processing
    # We'll treat all dimensions except 'dim' as batch dimension
    if dim != len(shape) - 1:
        # Permute so that dim is last
        perm = list(range(len(shape)))
        perm.pop(dim)
        perm.append(dim)
        x_perm = x.permute(perm)
        shape_perm = x_perm.shape
        # Now shape is (d0, ..., d_{dim-1}, d_{dim+1}, ..., dn, dim)
        # We want to process along last dimension
        x_flat = x_perm.reshape(-1, shape[dim])
        # Compute new strides
        stride_row = x_flat.stride(0)
        stride_col = x_flat.stride(1)
        out_shape = list(shape_perm[:-1])
    else:
        # Already last dimension, just flatten all but last
        x_flat = x.reshape(-1, shape[dim])
        stride_row = x_flat.stride(0)
        stride_col = x_flat.stride(1)
        out_shape = list(shape[:-1])
    
    n_rows = x_flat.shape[0]
    n_cols = shape[dim]
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=torch.long, device=x.device)
    
    # Set block size - tune based on n_cols
    BLOCK_SIZE = min(256, triton.next_power_of_2(n_cols))
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    argmin_kernel[grid](
        x_flat, out,
        n_rows, n_cols,
        stride_row, stride_col,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
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
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return triton_argmin(x, self.dim)