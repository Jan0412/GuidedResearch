import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_normalize_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (dimension per row)
    eps: tl.constexpr = 1e-12,  # Small value to avoid division by zero
    BLOCK_SIZE: tl.constexpr = 1024,  # Block size for parallelization
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute row start pointer
    row_start_ptr = x_ptr + row_idx * n_cols
    
    # Compute the L2 norm of the row
    # We accumulate in higher precision (float32) for numerical stability
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Process in blocks to handle large rows
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load values
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        
        # Convert to float32 and accumulate squared values
        x_f32 = x.to(tl.float32)
        acc += x_f32 * x_f32
    
    # Finalize norm computation (sqrt of sum)
    norm = tl.sqrt(acc + eps)
    
    # Normalize the row and store result
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load values
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        
        # Normalize and store
        x_normalized = x / norm
        tl.store(out_ptr + row_idx * n_cols + col_offsets, x_normalized, mask=mask)


def triton_l2_normalize(x: torch.Tensor):
    """
    Applies L2 normalization using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        
    Returns:
        torch.Tensor: L2-normalized tensor with same shape as input
    """
    # Ensure tensor is contiguous and on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()
    
    # Get dimensions
    batch_size, dim = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = (batch_size,)
    
    # Launch kernel
    l2_normalize_kernel[grid](
        x, 
        out, 
        batch_size, 
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return triton_l2_normalize(x)