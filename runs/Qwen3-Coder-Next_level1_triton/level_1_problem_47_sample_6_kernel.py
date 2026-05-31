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
    
    # Calculate row offset
    row_start = row_idx * n_cols
    
    # Initialize sum accumulator
    sum_acc = tl.zeros([1], dtype=tl.float32)
    
    # Process the row in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_acc = sum_acc + tl.sum(x, axis=0)
    
    # Store the result
    tl.store(out_ptr + row_idx, sum_acc)


def triton_sum_reduction(x: torch.Tensor, dim: int):
    """
    Performs sum reduction over the specified dimension using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (..., dim, ...)
        dim (int): Dimension to reduce over
        
    Returns:
        torch.Tensor: Output tensor after sum reduction with keepdim=True
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get original shape and compute reduced shape
    shape = list(x.shape)
    if dim < 0:
        dim = len(shape) + dim
    
    # Handle general case with arbitrary dimension
    # Reshape to 2D: (product of dims before dim, dim, product of dims after dim)
    if dim == 0:
        # Special case: reducing first dimension
        new_shape = (shape[0], -1)
        x_reshaped = x.view(shape[0], -1)
        n_rows = shape[0]
        n_cols = x_reshaped.size(1)
    else:
        # General case: move the reduction dimension to position 1
        perm = list(range(len(shape)))
        perm[0], perm[dim] = perm[dim], perm[0]
        x_permuted = x.permute(perm)
        
        # Reshape to 2D: (product of dims before reduction, reduction_dim, product of dims after)
        before_dim = 1
        after_dim = 1
        for i in range(dim):
            before_dim *= shape[i]
        for i in range(dim + 1, len(shape)):
            after_dim *= shape[i]
        
        x_reshaped = x_permuted.reshape(before_dim, shape[dim], -1)
        # For simplicity, collapse after_dim into the second dimension
        x_reshaped = x_reshaped.reshape(before_dim * after_dim, shape[dim])
        
        n_rows = before_dim * after_dim
        n_cols = shape[dim]
    
    # Prepare output tensor
    out_shape = shape.copy()
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # For sum reduction, we need to handle the output reshape
    if dim == 0:
        out = out.view(-1)
    else:
        # We need to handle the output correctly based on the permutation
        out_flat = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_rows,),)
    
    # Launch the Triton kernel
    sum_reduction_kernel[grid](x_reshaped, out_flat if dim != 0 else out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape output to match expected output shape
    if dim == 0:
        return out.view(*out_shape)
    else:
        # We need to handle the reshape correctly
        result = out_flat.view(*out_shape)
        return result


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