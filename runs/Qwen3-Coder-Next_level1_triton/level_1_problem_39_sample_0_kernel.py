import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_normalize_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (dim)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate row offset
    row_start = row_idx * n_cols
    
    # Compute sum of squares
    sum_sq = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute L2 norm
    norm = tl.sqrt(sum_sq)
    
    # Normalize and store
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        normalized = x / (norm + 1e-12)  # Add small epsilon for numerical stability
        tl.store(out_ptr + row_start + offsets, normalized, mask=mask)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L2 normalization along dimension 1 using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
    
    Returns:
        Output tensor with L2 normalization applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dim() == 2, "Input must be 2D tensor."
    
    # Ensure contiguous
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Tunable parameters
    BLOCK_SIZE = 512  # Power of 2 for efficiency
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    l2_normalize_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
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
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
        
        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return l2_normalize(x)