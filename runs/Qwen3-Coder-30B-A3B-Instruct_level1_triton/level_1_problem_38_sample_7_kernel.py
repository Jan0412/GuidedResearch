import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_mean_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Reduce to compute sum of absolute values
    sum_abs = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        abs_vals = tl.abs(x_vals)
        sum_abs += abs_vals
    
    # Sum all elements in the reduction buffer
    sum_abs = tl.sum(sum_abs, axis=0)
    
    # Compute mean (sum / n_cols)
    mean_val = sum_abs / n_cols
    
    # Store result
    tl.store(out_ptr + row_idx, mean_val)

@triton.jit
def l1_norm_divide_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Load mean value for this row
    mean_val = tl.load(mean_ptr + row_idx)
    
    # Avoid division by zero
    mean_val = tl.where(mean_val == 0.0, 1.0, mean_val)
    
    # Load and divide elements
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x_vals = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        result = x_vals / mean_val
        tl.store(out_ptr + row_start + cols, result, mask=mask)

def triton_l1_norm(x: torch.Tensor):
    """
    Triton-based L1 normalization implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # First kernel: compute means
    means = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE = 1024
    grid_mean = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    l1_norm_mean_kernel[grid_mean](
        x, means, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: perform division
    output = torch.empty_like(x)
    grid_div = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    l1_norm_divide_kernel[grid_div](
        x, means, output, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return triton_l1_norm(x)