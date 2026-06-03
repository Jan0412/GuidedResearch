import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current row index
    row_idx = tl.program_id(0)
    
    # Compute base pointers for this row
    x_offset = row_idx * n_cols
    out_offset = row_idx * n_cols
    
    # Initialize running sum for this row
    cumsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load input values
        x = tl.load(x_ptr + x_offset + col_offsets, mask=mask, other=0.0)
        
        # Convert to float32 for accumulation
        x_f32 = x.to(tl.float32)
        
        # Compute cumulative sum
        cumsum = cumsum + x_f32
        
        # Store result
        tl.store(out_ptr + out_offset + col_offsets, cumsum.to(tl.float32), mask=mask)


def triton_cumsum(x: torch.Tensor, dim: int):
    """
    Computes cumulative sum along specified dimension using Triton kernel.
    Optimized for FP32 precision.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Convert negative dimension to positive
    if dim < 0:
        dim = x.dim() + dim
    
    # For dim=1 (most common case in the example)
    if dim == 1:
        n_rows = x.shape[0]
        n_cols = x.shape[1]
        out = torch.empty_like(x)
        
        # Use appropriate block size based on column count
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
        
        # Launch kernel: one block per row
        grid = lambda meta: (n_rows,)
        cumsum_kernel[grid](x, out, n_cols, n_rows, BLOCK_SIZE=BLOCK_SIZE)
        return out
    else:
        # For other dimensions, use PyTorch as fallback (can be extended for other cases)
        return torch.cumsum(x, dim=dim)


class ModelNew(nn.Module):
    """
    Optimized version of the Scan model using custom Triton kernel for cumulative sum.
    """

    def __init__(self, dim):
        """
        Initialize the optimized Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for the optimized Scan model, using Triton kernel for cumulative sum.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying cumulative sum along self.dim.
        """
        return triton_cumsum(x, self.dim)