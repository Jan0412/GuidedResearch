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
    
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Compute the L2 norm for this row
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process elements in chunks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create offsets for this chunk
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid elements
        mask = offsets < n_cols
        # Load elements
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        # Accumulate squared values
        sum_sq += x_vals * x_vals
    
    # Reduce across the block to compute sum of squares for this row
    sum_of_squares = tl.sum(sum_sq, axis=0)
    
    # Compute the norm (avoid division by zero)
    norm = tl.sqrt(sum_of_squares + 1e-12)
    
    # Normalize the row
    for i in range(0, n_cols, BLOCK_SIZE):
        # Create offsets for this chunk
        offsets = i + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid elements
        mask = offsets < n_cols
        # Load elements
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
        
        # Determine grid size
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