import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    stride_x_row,
    stride_x_col,
    BLOCK_SIZE: tl.constexpr,
    DIM_SIZE: tl.constexpr
):
    # Each program handles one row
    row_id = tl.program_id(0)
    
    # Calculate base offset for this row
    row_base = row_id * stride_x_row
    
    # Shared memory for storing min values and indices within a block
    shared_min_vals = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_min_indices = tl.shared_memory(dtype=tl.int32, shape=(BLOCK_SIZE,))
    
    # Initialize local min tracking
    local_min_val = tl.full([], float('inf'), dtype=tl.float32)
    local_min_idx = tl.full([], 0, dtype=tl.int32)
    
    # Process elements in chunks
    for col_start in range(0, DIM_SIZE, BLOCK_SIZE):
        # Calculate actual column offset
        col_offset = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < DIM_SIZE
        
        # Load data from global memory
        x_offsets = row_base + col_offset * stride_x_col
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=float('inf'))
        
        # Find minimum in this chunk
        chunk_min_val = tl.min(x_vals)
        chunk_min_idx = tl.argmin(x_vals)
        
        # Update overall minimum if needed
        if chunk_min_val < local_min_val:
            local_min_val = chunk_min_val
            local_min_idx = col_start + chunk_min_idx
            
    # Store final result for this row
    tl.store(out_ptr + row_id, local_min_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For argmin along last dimension, we can optimize using Triton
        if self.dim == len(x.shape) - 1:
            return self._triton_argmin_last_dim(x)
        else:
            # Fall back to PyTorch for other dimensions
            return torch.argmin(x, dim=self.dim)

    def _triton_argmin_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        batch_size = x.shape[0]
        dim_size = x.shape[-1]
        
        # Prepare output
        out = torch.empty(batch_size, dtype=torch.int32, device=x.device)
        
        # Set up kernel launch parameters
        BLOCK_SIZE = 256
        grid_size = batch_size
        
        # Launch kernel
        argmin_kernel[grid_size](
            x,
            out,
            x.numel(),
            x.stride(0),
            x.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            DIM_SIZE=dim_size
        )
        
        return out