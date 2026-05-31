import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    inp_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    # Calculate the starting column for this thread block
    col_start = tl.program_id(1) * BLOCK_SIZE
    # Create offsets for this block
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bounds access
    mask = offsets < n_cols
    
    # Load input data for this row
    row = tl.load(inp_ptr + row_idx * n_cols + offsets, mask=mask, other=-float('inf'))
    
    # Subtract max for numerical stability
    row_max = tl.max(row, axis=0)
    row = row - row_max
    
    # Compute exponentials
    row = tl.exp(row)
    
    # Compute sum of exponentials
    row_sum = tl.sum(row, axis=0)
    
    # Normalize to get softmax
    row = row / row_sum
    
    # Store the result
    tl.store(out_ptr + row_idx * n_cols + offsets, row, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and on GPU
        x = x.contiguous().cuda()
        
        batch_size, n_cols = x.shape
        
        # Allocate output tensor
        output = torch.empty_like(x)
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid dimensions
        grid = (
            batch_size,
            (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
        
        # Launch kernel
        softmax_kernel[grid](
            x,
            output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output