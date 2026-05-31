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
        
    # Compute L2 norm for this row
    row_start = row_idx * n_cols
    norm = 0.0
    
    # First pass: compute sum of squares
    for col in range(0, n_cols, BLOCK_SIZE):
        offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        norm += tl.sum(x_vals * x_vals)
    
    # Take square root
    norm = tl.sqrt(norm)
    
    # Avoid division by zero
    norm = tl.where(norm == 0.0, 1.0, norm)
    
    # Second pass: normalize
    for col in range(0, n_cols, BLOCK_SIZE):
        offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        out_vals = x_vals / norm
        tl.store(out_ptr + row_start + offsets, out_vals, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    """
    Triton implementation of L2 normalization along dimension 1
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    
    grid = lambda meta: (batch_size,)
    
    l2_norm_kernel[grid](x, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for L2 normalization.
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