import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (dim)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    
    # Accumulate sum of absolute values
    sum_abs = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        # Load values and compute absolute value
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        sum_abs += tl.sum(abs_x, axis=0)
    
    # Compute mean: sum / n_cols
    mean_val = sum_abs / n_cols
    
    # Normalize the row: x / mean
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        # Load values
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        # Normalize and store
        out = x / mean_val
        tl.store(out_ptr + row_start + col_offsets, out, mask=mask)


def triton_l1_norm(x: torch.Tensor):
    """
    Applies L1 normalization: x / mean(|x|, dim=1, keepdim=True)
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    if len(shape) < 2:
        # If input is 1D, treat as single row
        x = x.unsqueeze(0)
        n_rows, n_cols = x.shape
    else:
        n_rows = shape[0]
        n_cols = shape[1]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Block size for column processing
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    l1_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape back if we expanded dimensions
    if len(shape) < 2:
        out = out.squeeze(0)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using Triton kernel.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return triton_l1_norm(x)