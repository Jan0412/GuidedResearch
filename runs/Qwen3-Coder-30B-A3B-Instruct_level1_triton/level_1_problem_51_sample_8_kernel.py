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
    
    if row_idx >= n_rows:
        return
        
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Initialize max value and index
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process elements in chunks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create offsets for this chunk
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load input values
        vals = tl.load(input_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        
        # Find max in this chunk
        chunk_max = tl.max(vals)
        chunk_max_idx = tl.arg_max(vals)
        
        # Update global max
        new_max = tl.maximum(max_val, chunk_max)
        # Handle tie-breaking - prefer earlier indices
        tie_break = tl.where(chunk_max > max_val, 0, 1)
        new_idx = tl.where(chunk_max > max_val, chunk_max_idx, max_idx)
        
        max_val = new_max
        max_idx = new_idx
        
    # Store the result
    tl.store(output_ptr + row_idx, max_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 0:
            # For argmax along dimension 0, we need a different approach
            # Since Triton doesn't easily support reduction across rows,
            # we'll fall back to PyTorch's implementation for this case
            return torch.argmax(x, dim=self.dim)
        elif self.dim == 1:
            # For argmax along dimension 1, we can optimize with Triton
            # Reshape to 2D for easier processing
            original_shape = x.shape
            n_rows = original_shape[0]
            n_cols = original_shape[1] * original_shape[2]
            
            # Flatten the last two dimensions
            x_flat = x.view(n_rows, n_cols)
            
            # Prepare output
            output = torch.empty(n_rows, dtype=torch.int64, device=x.device)
            
            # Launch kernel
            BLOCK_SIZE = 1024
            grid = lambda meta: (n_rows,)
            
            argmax_kernel[grid](x_flat, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
            return output
        else:
            # For other dimensions, fall back to PyTorch
            return torch.argmax(x, dim=self.dim)