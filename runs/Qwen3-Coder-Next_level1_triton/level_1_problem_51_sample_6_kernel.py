import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (size of dimension to reduce)
    ROW_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate starting pointer for this row
    x_row_ptr = x_ptr + row_idx * n_cols
    
    # Initialize max value and index
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process the row in chunks
    for start_col in range(0, n_cols, ROW_SIZE):
        col_offsets = start_col + tl.arange(0, ROW_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=float('-inf'))
        
        # Compare with current max
        greater_mask = x > max_val
        max_val = tl.where(greater_mask, x, max_val)
        # Update index only where we found a new maximum
        new_idx = tl.where(greater_mask, col_offsets, max_idx)
        # For ties, we keep the first occurrence (lower index)
        equal_mask = (x == max_val) & (col_offsets < max_idx)
        max_idx = tl.where(equal_mask, col_offsets, new_idx)
    
    # Store the result
    tl.store(out_ptr + row_idx, max_idx)


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs argmax operation using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to perform argmax over
        
    Returns:
        Tensor with argmax indices along the specified dimension
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    n_dims = len(shape)
    
    # Normalize dimension to positive
    if dim < 0:
        dim += n_dims
    
    # Calculate sizes
    if dim == n_dims - 1:
        # Last dimension: easy case, process rows
        n_rows = 1
        for i in range(n_dims - 1):
            n_rows *= shape[i]
        n_cols = shape[dim]
        row_size = min(128, n_cols)  # Block size for processing each row
    else:
        # For other dimensions, we need to transpose the tensor
        # Move the target dimension to the end
        dims = list(range(n_dims))
        dims.pop(dim)
        dims.append(dim)
        x = x.permute(dims).contiguous()
        
        # After permutation, the target dimension is last
        shape = x.shape
        n_rows = 1
        for i in range(n_dims - 1):
            n_rows *= shape[i]
        n_cols = shape[-1]
        row_size = min(128, n_cols)
    
    # Prepare output tensor
    out_shape = list(shape[:-1])
    out = torch.empty(out_shape, dtype=torch.long, device=x.device)
    
    # Launch kernel with one block per row
    grid = (n_rows,)
    
    # Launch the Triton kernel
    argmax_kernel[grid](x, out, n_rows, n_cols, ROW_SIZE=row_size)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return triton_argmax(x, self.dim)