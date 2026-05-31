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
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_start = row_idx * n_cols
    
    # Initialize accumulator
    sum_val = tl.zeros((1,), dtype=tl.float32)
    
    # Process columns in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_indices, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_val = sum_val + tl.sum(x, axis=0)
    
    # Store result
    tl.store(out_ptr + row_idx, sum_val)


def triton_sum_reduce(x: torch.Tensor, dim: int):
    """
    Perform sum reduction over specified dimension using Triton kernel.
    Supports only keepdim=True (as per original torch.sum implementation).
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()
    
    # Get shape
    shape = x.shape
    ndim = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim = ndim + dim
    
    # Handle the case where dim is the last dimension for simplicity in kernel design
    # We'll permute dimensions if needed to make the reduction dimension the last one
    if dim != ndim - 1:
        # Create permutation to move reduction dimension to the end
        perm = list(range(ndim))
        perm.pop(dim)
        perm.append(dim)
        
        # Permute input
        x_permuted = x.permute(perm)
        shape_permuted = x_permuted.shape
        
        # Get product of all dimensions except last (reduction dimension)
        n_rows = 1
        for i in shape_permuted[:-1]:
            n_rows *= i
        n_cols = shape_permuted[-1]
        
        # Flatten to 2D
        x_flat = x_permuted.reshape(n_rows, n_cols)
        
        # Prepare output
        out_flat = torch.empty((n_rows, 1), dtype=x.dtype, device=x.device)
        
        # Launch kernel
        BLOCK_SIZE = 256
        grid = lambda meta: (n_rows,)
        sum_reduction_kernel[grid](x_flat, out_flat, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        
        # Reshape output back to original shape with reduced dimension
        out_shape = list(shape)
        out_shape[dim] = 1
        out = out_flat.reshape(out_shape)
    else:
        # Simple case: reduction is on last dimension
        n_rows = 1
        for i in shape[:-1]:
            n_rows *= i
        n_cols = shape[-1]
        
        x_flat = x.reshape(n_rows, n_cols)
        out_flat = torch.empty((n_rows, 1), dtype=x.dtype, device=x.device)
        
        BLOCK_SIZE = 256
        grid = lambda meta: (n_rows,)
        sum_reduction_kernel[grid](x_flat, out_flat, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        
        # Reshape output
        out_shape = list(shape)
        out_shape[dim] = 1
        out = out_flat.reshape(out_shape)
    
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
        return triton_sum_reduce(x, self.dim)