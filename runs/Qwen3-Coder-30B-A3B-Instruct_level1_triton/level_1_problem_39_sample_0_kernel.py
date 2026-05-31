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
    sum_sq = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offset and limit
        col_offset = i
        actual_limit = min(BLOCK_SIZE, n_cols - i)
        
        # Load elements with masking
        offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate squared values
        sum_sq += tl.sum(x_vals * x_vals)
    
    # Compute square root of sum of squares
    norm = tl.sqrt(sum_sq)
    
    # Avoid division by zero
    norm = tl.maximum(norm, 1e-12)
    
    # Normalize the row
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offset = i
        actual_limit = min(BLOCK_SIZE, n_cols - i)
        
        # Load elements with masking
        offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Divide by norm and store
        normalized = x_vals / norm
        tl.store(out_ptr + offsets, normalized, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using Triton kernels.
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
        # Ensure input is contiguous and on GPU
        x = x.contiguous().cuda()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        batch_size, dim = x.shape
        
        # Configure block size
        BLOCK_SIZE = 1024
        
        # Calculate grid size
        grid = (batch_size,)
        
        # Launch Triton kernel
        l2_norm_kernel[grid](
            x,
            out,
            batch_size,
            dim,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out