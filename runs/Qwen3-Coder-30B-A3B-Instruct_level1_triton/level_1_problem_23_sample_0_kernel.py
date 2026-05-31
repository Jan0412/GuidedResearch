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
    # Create offsets for accessing elements
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask for valid columns
    mask = offsets < n_cols
    
    # Load input data for this row
    input_row = tl.load(input_ptr + row_idx * n_cols + offsets, mask=mask, other=-float('inf'))
    
    # Numerically stable softmax: subtract max value
    row_max = tl.max(input_row, axis=0)
    input_row = input_row - row_max
    
    # Compute exp
    exp_row = tl.exp(input_row)
    
    # Compute sum of exponentials
    exp_sum = tl.sum(exp_row, axis=0)
    
    # Compute softmax
    softmax_row = exp_row / exp_sum
    
    # Store result
    tl.store(output_ptr + row_idx * n_cols + offsets, softmax_row, mask=mask)

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