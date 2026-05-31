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
    # Get the row index this program instance handles
    row_idx = tl.program_id(0)
    
    # Compute the base pointer for this row
    row_start = row_idx * n_cols
    
    # Compute the L2 norm for this row
    norm_sq = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        norm_sq += tl.sum(x * x, axis=0)
    
    # Compute L2 norm
    norm = tl.sqrt(norm_sq)
    
    # Normalize and store the result
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        out = x / (norm + 1e-12)  # Add small epsilon for numerical stability
        tl.store(out_ptr + row_start + cols, out, mask=mask)


def triton_l2_norm(x: torch.Tensor):
    """
    Applies L2 normalization along dimension 1 using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        
    Returns:
        torch.Tensor: L2 normalized tensor with same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    assert x.dim() == 2, "Input must be 2D tensor"
    
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    
    # Use a reasonable block size for the reduction
    BLOCK_SIZE = 512
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the kernel
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
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
            
        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return triton_l2_norm(x)