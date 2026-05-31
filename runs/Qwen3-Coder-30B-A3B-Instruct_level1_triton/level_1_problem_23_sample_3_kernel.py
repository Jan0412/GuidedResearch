import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    # Calculate the starting column for this thread block
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # Load input values for this row
    input_ptrs = input_ptr + row_idx * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = (col_start + tl.arange(0, BLOCK_SIZE)) < n_cols
    
    # Load input values with masking
    input_vals = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute max value for numerical stability
    max_val = tl.max(input_vals, axis=0)
    
    # Subtract max for numerical stability
    input_vals = input_vals - max_val
    
    # Compute exp
    exp_vals = tl.exp(input_vals)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    softmax_vals = exp_vals / sum_exp
    
    # Store results
    output_ptrs = output_ptr + row_idx * n_cols + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptrs, softmax_vals, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Create output tensor
        output = torch.empty_like(x)
        
        # Get dimensions
        batch_size, n_cols = x.shape
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid dimensions
        grid_x = batch_size
        grid_y = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (grid_x, grid_y)
        
        # Launch kernel
        softmax_kernel[grid](
            x,
            output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output