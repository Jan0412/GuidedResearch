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
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Calculate starting position for this row
    row_start = row_idx * n_cols
    
    # Compute L2 norm for this row
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        sum_sq += x_vals * x_vals
    
    # Reduce sum of squares across all columns in this row
    sum_of_squares = tl.sum(sum_sq, axis=0)
    norm = tl.sqrt(sum_of_squares)
    
    # Avoid division by zero
    norm = tl.where(norm == 0.0, 1.0, norm)
    
    # Normalize the row
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        out_vals = x_vals / norm
        tl.store(out_ptr + row_start + cols, out_vals, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    """
    Triton-based L2 normalization implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Use a reasonable block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (batch_size, 1, 1)
    
    # Launch the Triton kernel
    l2_norm_kernel[grid](
        x,
        out,
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for L2 normalization.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (*, dim, *).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return triton_l2_norm(x)