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
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for sum reduction along the last dimension.
    Each program handles one row of the input tensor.
    """
    # Get the row index this program instance processes
    row_idx = tl.program_id(0)
    
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Initialize accumulator
    sum = tl.zeros([1], dtype=tl.float32)
    
    # Process the row in chunks
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data with mask
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        sum += x
    
    # Store the result
    tl.store(out_ptr + row_idx, sum)


def triton_sum_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Perform sum reduction along the specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Output tensor with the specified dimension reduced to size 1
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get the shape and adjust for negative dimension indexing
    shape = list(x.shape)
    if dim < 0:
        dim = len(shape) + dim
    
    # Calculate dimensions for the kernel
    # We'll flatten everything before and after the target dimension
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    
    n_cols = shape[dim]
    out_shape = shape[:dim] + [1] + shape[dim+1:] if dim < len(shape) - 1 else [1]
    
    # Handle the case where we're reducing the last dimension
    if dim == len(shape) - 1:
        n_rows = x.numel() // n_cols
        out_shape = shape[:dim] + [1]
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Determine block size (power of 2 for efficiency)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    
    # Grid configuration: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    sum_reduction_kernel[grid](
        x, out, n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension
    using custom Triton kernel.
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
        return triton_sum_reduce(x, self.dim)