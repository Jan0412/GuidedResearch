import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate start pointer for this row
    row_start = row_idx * n_cols
    
    # Accumulator for sum of squares
    sum_squares = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over columns in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Accumulate sum of squares
        sum_squares += tl.sum(x * x, axis=0)
    
    # Compute norm
    norm = tl.sqrt(sum_squares)
    
    # Normalize and store results
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = x / norm
        
        # Store result
        tl.store(out_ptr + row_start + col_offsets, normalized, mask=mask)


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L2 normalization along dimension 1 using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    assert x.dim() >= 2, "Input must have at least 2 dimensions."
    assert x.size(1) > 0, "Dimension 1 must be non-empty."
    
    x = x.contiguous()
    
    # Get dimensions
    n_rows = x.size(0)
    n_cols = x.size(1)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine block size (tuned for FP32 and memory coalescing)
    BLOCK_SIZE = 256
    
    # Grid: one block per row
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    l2_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using optimized Triton kernel.
        """
        return triton_l2_norm(x)