import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Shared memory for reduction
    shared_min_val = tl.shared_ptr(tl.float32, BLOCK_SIZE)
    shared_min_idx = tl.shared_ptr(tl.int32, BLOCK_SIZE)
    
    # Initialize local variables
    local_min_val = tl.full([], float('inf'), dtype=tl.float32)
    local_min_idx = tl.full([], 0, dtype=tl.int32)
    
    # Process elements in chunks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual column offset
        col_offset = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < n_cols
        
        # Load data
        x_vals = tl.load(x_ptr + row_idx * n_cols + col_offset, mask=mask, other=float('inf'))
        
        # Find min in this chunk
        chunk_min_val = tl.min(x_vals)
        chunk_min_idx = tl.argmin(x_vals)
        
        # Update global min
        if chunk_min_val < local_min_val:
            local_min_val = chunk_min_val
            local_min_idx = col_start + chunk_min_idx
            
    # Store result
    tl.store(output_ptr + row_idx, local_min_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            # For non-dimension 1, fall back to PyTorch implementation
            return torch.argmin(x, dim=self.dim)
            
        # For dimension 1, use Triton kernel
        assert x.dim() == 3, "Expected 3D tensor"
        batch_size, dim1, dim2 = x.shape
        
        # Ensure tensor is contiguous and on GPU
        x = x.contiguous().cuda()
        
        # Prepare output
        output = torch.zeros(batch_size, dtype=torch.int32, device=x.device)
        
        # Kernel launch parameters
        BLOCK_SIZE = 1024
        grid_size = (batch_size,)
        
        # Launch kernel
        argmin_kernel[grid_size](
            x,
            output,
            batch_size,
            dim2,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output