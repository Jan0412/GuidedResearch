import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def l2_norm_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_cols,  # Number of columns (dim) per row
    n_rows,  # Number of rows (batch_size)
    eps: tl.constexpr = 1e-8,  # Small epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr = 1024,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate pointer offsets for this row
    row_start_ptr = x_ptr + row_idx * n_cols
    out_row_start_ptr = out_ptr + row_idx * n_cols
    
    # Accumulate squared sum
    acc_sumsq = tl.zeros([1], dtype=tl.float32)
    
    # Process in chunks to handle large dimensions with limited block size
    num_blocks = tl.cdiv(n_cols, BLOCK_SIZE)
    
    for block_idx in range(num_blocks):
        col_offset = block_idx * BLOCK_SIZE
        offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        
        # Convert to float32 for accumulation to avoid precision loss
        x_f32 = x.to(tl.float32)
        acc_sumsq += tl.sum(x_f32 * x_f32)
    
    # Compute L2 norm (sqrt of sum of squares)
    norm = tl.sqrt(acc_sumsq)
    
    # Avoid division by zero
    inv_norm = tl.where(norm > eps, 1.0 / norm, 0.0)
    
    # Normalize and store result
    for block_idx in range(num_blocks):
        col_offset = block_idx * BLOCK_SIZE
        offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data again
        x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = x * inv_norm
        
        # Store result
        tl.store(out_row_start_ptr + offsets, normalized.to(x_ptr.dtype.element_ty), mask=mask)


def triton_l2_norm(x: torch.Tensor):
    """
    Applies L2 normalization to the input tensor using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        
    Returns:
        torch.Tensor: L2 normalized tensor with same shape as input
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dim() == 2, "Input must be 2D tensor"
    
    # Ensure contiguous memory layout
    x = x.contiguous()
    
    # Create output tensor with same shape and dtype
    out = torch.empty_like(x)
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    # Launch kernel
    l2_norm_kernel[grid](
        x, out, n_cols, n_rows,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
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
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return triton_l2_norm(x)