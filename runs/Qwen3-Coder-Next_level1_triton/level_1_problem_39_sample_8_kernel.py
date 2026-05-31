import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,           # Pointer to input tensor
    out_ptr,         # Pointer to output tensor
    n_rows,          # Number of rows
    n_cols,          # Number of columns (dimension along which to normalize)
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr = 1e-12
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate base pointers for this row
    x_row_start = x_ptr + row_idx * n_cols
    out_row_start = out_ptr + row_idx * n_cols
    
    # Compute L2 norm using online computation to avoid overflow
    # First pass: compute sum of squares
    sum_sq = tl.zeros([1], dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    for start_col in range(0, n_cols, BLOCK_SIZE):
        mask = (col_offsets < n_cols - start_col)
        offsets = start_col + col_offsets
        x_val = tl.load(x_row_start + offsets, mask=mask, other=0.0)
        x_val_f32 = x_val.to(tl.float32)
        sum_sq += x_val_f32 * x_val_f32
    
    # Compute square root of sum of squares
    norm = tl.sqrt(sum_sq)
    
    # Second pass: normalize elements
    for start_col in range(0, n_cols, BLOCK_SIZE):
        mask = (col_offsets < n_cols - start_col)
        offsets = start_col + col_offsets
        x_val = tl.load(x_row_start + offsets, mask=mask, other=0.0)
        norm_val = norm + EPS if norm == 0.0 else norm
        out_val = x_val / norm_val
        tl.store(out_row_start + offsets, out_val, mask=mask)


def triton_l2_norm(x: torch.Tensor):
    """
    Applies L2 normalization along dimension 1 using Triton kernel.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
        
    Returns:
        Output tensor with L2 normalization applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dim() == 2, "Input must be 2D tensor"
    
    x = x.contiguous()
    batch_size, dim = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Grid: one block per row
    grid = (batch_size,)
    
    # Launch kernel
    l2_norm_kernel[grid](
        x, out, batch_size, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using Triton kernel.
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