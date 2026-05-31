import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Early exit if row index is out of bounds
    if row_idx >= n_rows:
        return
    
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Initialize max value and index
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process the row in chunks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual column indices
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load values from memory
        vals = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=float('-inf'))
        
        # Find max in current chunk
        chunk_max_val = tl.max(vals)
        chunk_max_idx = tl.arg_max(vals)
        
        # Update global max if needed
        update_mask = chunk_max_val > max_val
        max_val = tl.where(update_mask, chunk_max_val, max_val)
        max_idx = tl.where(update_mask, chunk_max_idx + col_start, max_idx)
    
    # Store result
    tl.store(output_ptr + row_idx, max_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 2:
            # For argmax along last dimension
            n_rows = x.shape[0] * x.shape[1]
            n_cols = x.shape[2]
            
            # Ensure input is contiguous
            x = x.contiguous()
            
            # Prepare output tensor
            output = torch.empty(n_rows, dtype=torch.int64, device=x.device)
            
            # Set up kernel launch parameters
            BLOCK_SIZE = 1024
            
            # Grid size calculation
            grid = lambda meta: (n_rows,)
            
            # Launch kernel
            argmax_kernel[grid](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
            
            # Reshape output to match expected dimensions
            return output.view(x.shape[0], x.shape[1])
        else:
            # Fall back to PyTorch implementation for other dimensions
            return torch.argmax(x, dim=self.dim)