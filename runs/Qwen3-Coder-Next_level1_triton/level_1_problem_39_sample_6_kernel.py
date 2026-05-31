import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (dim)
    BLOCK_SIZE: tl.constexpr,
):
    # Compute row index
    row_idx = tl.program_id(0)
    
    # Start pointer for this row
    row_start = row_idx * n_cols
    
    # Compute squared sum for this row using tiling
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over columns in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Accumulate squared values
        acc += x * x
    
    # Reduce within the block (though here acc is just a scalar per row)
    # Since we're doing a single block reduction per row, we just need to compute sqrt
    norm = tl.sqrt(acc)
    
    # Avoid division by zero - add small epsilon to norm
    norm = tl.where(norm > 0, norm, 1.0)
    
    # Second pass: normalize the data
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Normalize and store
        normalized = x / norm
        tl.store(out_ptr + row_start + col_offsets, normalized, mask=mask)


def triton_l2_norm(x: torch.Tensor):
    """
    Applies L2 normalization to the input tensor along dimension 1.
    
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
    
    # Choose block size - for wide matrices, we want to process columns in chunks
    # 1024 is a good balance for the wide dimension (65535)
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the Triton kernel
    l2_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using Triton kernels.
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