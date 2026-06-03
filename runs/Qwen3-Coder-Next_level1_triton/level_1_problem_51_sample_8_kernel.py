import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate which row this program instance handles
    row_idx = tl.program_id(0)
    
    # Calculate the offset to start of this row
    row_start = row_idx * n_cols if dim == 1 else row_idx
    
    # Initialize max value and index
    max_val = -float("inf")
    max_idx = 0
    
    # Process in blocks to find max and its index
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        offsets = row_start + col_offsets if dim == 1 else row_start + col_offsets * n_rows
        x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
        
        # Check if any value is greater than current max
        greater_mask = x > max_val
        # Get the first true index in the mask
        # We use a simple approach: check each element
        for i in range(BLOCK_SIZE):
            if i < BLOCK_SIZE and tl.load(mask + i) and tl.load(greater_mask + i):
                val = tl.load(x + i)
                if val > max_val:
                    max_val = val
                    max_idx = col_start + i
    
    # Store result
    tl.store(out_ptr + row_idx, max_idx.to(tl.int64))


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton-based argmax implementation.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute argmax
        
    Returns:
        Tensor with argmax indices along the specified dimension
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    shape = x.shape
    n_dims = len(shape)
    
    # Normalize negative dimension
    if dim < 0:
        dim = n_dims + dim
    
    # Handle the dimension to reduce
    if dim == n_dims - 1:
        # Last dimension - process rows
        n_rows = 1
        for i in range(n_dims - 1):
            n_rows *= shape[i]
        n_cols = shape[dim]
        out_shape = shape[:dim]
    else:
        # Need to transpose to make dim the last dimension
        dims = list(range(n_dims))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        x = x.permute(dims).contiguous()
        new_shape = x.shape
        n_rows = 1
        for i in range(n_dims - 1):
            n_rows *= new_shape[i]
        n_cols = new_shape[-1]
        out_shape = new_shape[:-1]
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=torch.long, device=x.device)
    
    # Determine block size based on dimension size
    BLOCK_SIZE = 128
    if n_cols <= 128:
        BLOCK_SIZE = 64
    if n_cols <= 64:
        BLOCK_SIZE = 32
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    argmax_kernel[grid](x, out, n_rows, n_cols, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax using Triton kernel.
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