import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output indices pointer
    n_rows,  # Number of rows (batch elements)
    n_cols,  # Number of columns (elements per row)
    stride_row,  # Stride between rows
    stride_col,  # Stride between columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute starting pointer for this row
    x_row_ptr = x_ptr + row_idx * stride_row
    
    # Initialize min value and index
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    min_idx = tl.full((BLOCK_SIZE,), 0, dtype=tl.int32)
    
    # Process in blocks to find minimum in the row
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load values
        vals = tl.load(x_row_ptr + offsets * stride_col, mask=mask, other=float('inf'))
        
        # Compare with current minimum
        is_smaller = vals < min_val
        min_val = tl.where(is_smaller, vals, min_val)
        min_idx = tl.where(is_smaller, offsets, min_idx)
    
    # Reduce across blocks to find the global minimum
    block_idx = tl.arange(0, BLOCK_SIZE)
    block_mask = block_idx < n_cols
    block_min_val = tl.min(min_val, axis=0, mask=block_mask)
    
    # Find the first index with the minimum value
    final_idx = 0
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(x_row_ptr + offsets * stride_col, mask=mask, other=float('inf'))
        is_equal = vals == block_min_val
        found_idx = tl.where(is_equal, offsets, n_cols)
        block_min_idx = tl.min(found_idx, axis=0, mask=block_mask)
        if block_min_idx < n_cols:
            final_idx = block_min_idx
            break
    
    # Store the result
    tl.store(out_ptr + row_idx, final_idx)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of argmin along a specified dimension.
    Optimized for FP32 tensors.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    
    # Ensure tensor is contiguous
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    dim = dim if dim >= 0 else len(shape) + dim
    
    # Compute strides
    stride = x.stride()
    
    # Determine which dimension to process
    if dim == len(shape) - 1:
        # Last dimension: straightforward case
        n_rows = 1
        for s in shape[:-1]:
            n_rows *= s
        n_cols = shape[dim]
        stride_row = stride[dim]
        stride_col = 1
    else:
        # Need to permute dimensions to make the target dimension last
        # Move the target dimension to the end
        dims = list(range(len(shape)))
        dims.pop(dim)
        dims.append(dim)
        x = x.permute(dims).contiguous()
        new_shape = x.shape
        n_rows = 1
        for s in new_shape[:-1]:
            n_rows *= s
        n_cols = new_shape[-1]
        stride_row = x.stride(-1)
        stride_col = 1
    
    # Prepare output tensor
    out_shape = list(shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=torch.long, device=x.device)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Compute grid
    grid = (n_rows,)
    
    # Launch kernel
    argmin_kernel[grid](
        x, out, n_rows, n_cols, stride_row, stride_col,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for argmin operation.
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