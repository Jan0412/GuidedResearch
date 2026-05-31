import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (dimension)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate the starting pointer for this row
    row_start = row_idx * n_cols
    
    # Accumulator for the squared sum
    acc_sumsq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process columns in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < n_cols
        
        # Load elements
        x = tl.load(x_ptr + row_start + col_indices, mask=mask, other=0.0)
        
        # Accumulate squared sum
        acc_sumsq += x * x
    
    # Reduce within the block
    sumsq = tl.sum(acc_sumsq, axis=0)
    
    # Compute norm with numerical stability
    norm = tl.sqrt(sumsq)
    
    # Avoid division by zero
    norm = tl.where(norm > 1e-12, norm, 1.0)
    
    # Process columns again for normalization
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_indices < n_cols
        
        # Load elements
        x = tl.load(x_ptr + row_start + col_indices, mask=mask, other=0.0)
        
        # Normalize and store
        out = x / norm
        tl.store(out_ptr + row_start + col_indices, out, mask=mask)


def triton_l2_norm(x: torch.Tensor):
    """
    Applies L2 normalization to the input tensor along dimension 1.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        
    Returns:
        torch.Tensor: L2 normalized tensor with same shape as input
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dim() == 2, "Input must be 2D tensor"
    
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    
    # Use a reasonable block size
    BLOCK_SIZE = 256
    
    # Grid: one program per row
    grid = (n_rows,)
    
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
        Applies L2 normalization to the input tensor using Triton kernel.
        """
        return triton_l2_norm(x)