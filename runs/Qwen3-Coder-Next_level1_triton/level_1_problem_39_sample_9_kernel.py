import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def l2_norm_kernel(
    x_ptr,      # Pointer to input tensor
    out_ptr,    # Pointer to output tensor
    n_rows,     # Number of rows
    n_cols,     # Number of columns (dim)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Accumulator for squared sum
    acc_sumsq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Loop over columns in chunks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + col_offsets
        mask = offsets < n_cols
        # Load data
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        # Accumulate squared values
        acc_sumsq += x.to(tl.float32) * x.to(tl.float32)
    
    # Reduce within block to get sum of squares for this row
    # Since we're doing row-wise normalization and each program handles one row,
    # we can sum across the block dimension
    sumsq = tl.sum(acc_sumsq, axis=0)
    
    # Compute norm with small epsilon for numerical stability
    norm = tl.sqrt(sumsq + 1e-12)
    
    # Normalize and store
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + col_offsets
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        normalized = x / norm
        tl.store(out_ptr + row_start + offsets, normalized, mask=mask)


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L2 normalization along dimension 1 (column dimension) using Triton kernel.
    Assumes x is 2D tensor of shape (batch_size, dim).
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    assert x.dim() == 2, "Input must be 2D tensor."
    x = x.contiguous()
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Determine block size (use a reasonable size for good occupancy)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    
    # Grid: one program per row
    grid = (n_rows,)
    
    # Launch kernel
    l2_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        return triton_l2_norm(x)