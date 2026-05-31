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
    # Calculate the starting column index for this thread block
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # Load input data for this row
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load input values
    inp_vals = tl.load(inp_ptr + row_idx * n_cols + offsets, mask=mask, other=-float('inf'))
    
    # Subtract max for numerical stability
    max_val = tl.max(inp_vals, axis=0)
    inp_vals = inp_vals - max_val
    
    # Compute exponentials
    exp_vals = tl.exp(inp_vals)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    softmax_vals = exp_vals / sum_exp
    
    # Store results
    tl.store(out_ptr + row_idx * n_cols + offsets, softmax_vals, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        batch_size, n_cols = x.shape
        
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
            out,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out