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
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Check if we're within bounds
    if row_idx >= n_rows:
        return
    
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Compute L2 norm for this row
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # Create mask
        mask = offsets < n_cols
        
        # Load data
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Accumulate squared values
        sum_sq += x_vals * x_vals
    
    # Reduce across the block to get sum of squares for this row
    sum_of_squares = tl.sum(sum_sq, axis=0)
    
    # Compute L2 norm (sqrt of sum of squares)
    l2_norm = tl.sqrt(sum_of_squares)
    
    # Avoid division by zero
    l2_norm = tl.where(l2_norm == 0.0, 1.0, l2_norm)
    
    # Normalize the row
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # Create mask
        mask = offsets < n_cols
        
        # Load original values
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Divide by L2 norm
        normalized_vals = x_vals / l2_norm
        
        # Store result
        tl.store(out_ptr + row_start + offsets, normalized_vals, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    """
    Apply L2 normalization using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid size
    grid = (n_rows,)
    
    # Launch kernel
    l2_norm_kernel[grid](
        x,
        out,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using Triton kernels.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.
        """
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