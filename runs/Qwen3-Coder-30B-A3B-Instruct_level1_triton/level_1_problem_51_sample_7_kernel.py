import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Check if row index is valid
    if row_idx >= n_rows:
        return
    
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Shared memory for reduction
    shared_max = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_idx = tl.shared_memory(dtype=tl.int32, shape=(BLOCK_SIZE,))
    
    # Initialize max and index
    max_val = tl.full([], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([], 0, dtype=tl.int32)
    
    # Process the entire row in chunks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual column offset
        col_offset = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < n_cols
        
        # Load data from global memory
        x_vals = tl.load(x_ptr + row_start + col_offset, mask=mask, other=-float('inf'))
        
        # Find max in this chunk
        chunk_max = tl.max(x_vals)
        chunk_idx = tl.argmax(x_vals)
        
        # Update global max and index
        if chunk_max > max_val:
            max_val = chunk_max
            max_idx = chunk_idx + col_start
    
    # Store result
    tl.store(out_ptr + row_idx, max_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 2:  # Only optimize for dim=2 case
            return torch.argmax(x, dim=self.dim)
        
        # For dim=2 case, we need to compute argmax along last dimension
        batch_size, dim1, dim2 = x.shape
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        out = torch.empty(batch_size, dim1, dtype=torch.int64, device=x.device)
        
        # Grid configuration
        grid_size = batch_size * dim1
        BLOCK_SIZE = 1024
        
        # Launch kernel
        argmax_kernel[grid_size](
            x,
            out,
            batch_size,
            dim2,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out