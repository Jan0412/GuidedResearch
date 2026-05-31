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
    
    # Check if this row is within bounds
    if row_idx >= n_rows:
        return
    
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Compute L2 norm for this row
    sum_sq = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create offsets for this chunk
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # Create mask to avoid going out of bounds
        mask = offsets < n_cols
        
        # Load data
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Accumulate squared values
        sum_sq += tl.sum(x_vals * x_vals)
    
    # Compute square root of sum of squares
    norm = tl.sqrt(sum_sq)
    
    # Avoid division by zero
    norm = tl.where(norm == 0.0, 1.0, norm)
    
    # Normalize the row
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create offsets for this chunk
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # Create mask to avoid going out of bounds
        mask = offsets < n_cols
        
        # Load data
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Normalize and store
        normalized = x_vals / norm
        tl.store(out_ptr + row_start + offsets, normalized, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and on GPU
        x = x.contiguous().cuda()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        n_rows, n_cols = x.shape
        
        # Configure block size
        BLOCK_SIZE = 1024
        
        # Calculate grid size
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