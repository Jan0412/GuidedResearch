import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one row
    row_idx = tl.program_id(0)
    
    # Pointer to the start of the current row
    x_row_ptr = x_ptr + row_idx * row_stride
    
    # Initialize min value and index
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    min_idx = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    
    # Process columns in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load values
        vals = tl.load(x_row_ptr + col_offsets, mask=mask, other=float('inf'))
        
        # Create mask for comparison
        new_min_mask = vals < min_val
        
        # Update minimum values and indices where new minimum found
        min_val = tl.where(new_min_mask, vals, min_val)
        min_idx = tl.where(new_min_mask, col_offsets, min_idx)
    
    # Find the overall minimum across the block
    # Use tl.min to get the minimum value and tl.argmin to get its index
    final_min_val = tl.min(min_val)
    # Get the index of the minimum value in the min_val array
    idx_in_block = tl.argmax(min_val, axis=0)
    
    # Get the actual column index
    final_idx = tl.load(min_idx + idx_in_block)
    
    # Store result
    tl.store(out_ptr + row_idx, final_idx)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of argmin along a specified dimension.
    
    Args:
        x: Input tensor (should be contiguous)
        dim: Dimension along which to compute argmin
        
    Returns:
        Tensor with indices of minimum values along the specified dimension
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get shape information
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
        
    # Calculate dimensions for kernel
    if dim == len(shape) - 1:
        # Last dimension: easy case
        n_rows = 1
        for s in shape[:-1]:
            n_rows *= s
        n_cols = shape[-1]
        row_stride = shape[-1]
    elif dim == 0:
        # First dimension: need to transpose view
        # Reshape to (n_cols, n_rows) where n_cols = shape[0], n_rows = product of other dims
        n_rows = 1
        for s in shape[1:]:
            n_rows *= s
        n_cols = shape[0]
        row_stride = n_rows
    else:
        # Middle dimension: need to permute
        # Move dim to last position, then apply same logic as for last dimension
        perm = list(range(len(shape)))
        perm.pop(dim)
        perm.append(dim)
        x_permuted = x.permute(perm)
        shape_perm = x_permuted.shape
        n_rows = 1
        for s in shape_perm[:-1]:
            n_rows *= s
        n_cols = shape_perm[-1]
        row_stride = shape_perm[-1]
    
    # Ensure BLOCK_SIZE is at least as large as n_cols
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Cap at reasonable size
    
    # Prepare output tensor
    out_shape = list(shape)
    del out_shape[dim]
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    
    # Launch kernel
    grid = lambda meta: (n_rows,)
    argmin_kernel[grid](x, out, n_rows, n_cols, row_stride, BLOCK_SIZE=BLOCK_SIZE)
    
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