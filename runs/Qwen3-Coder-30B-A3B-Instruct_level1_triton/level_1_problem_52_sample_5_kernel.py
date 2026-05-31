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
    # Get the row (batch) index
    row = tl.program_id(0)
    
    # Each program handles one row
    block_start = row * stride_x_row
    
    # Shared memory for reduction
    shared_min_vals = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_min_indices = tl.shared_memory(dtype=tl.int32, shape=(BLOCK_SIZE,))
    
    # Initialize local min values and indices
    local_min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    local_min_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
    
    # Loop over columns in chunks
    for col_start in range(0, DIM_SIZE, BLOCK_SIZE):
        # Calculate actual column offset
        col_offset = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < DIM_SIZE
        
        # Load data from global memory
        x_offsets = block_start + col_offset * stride_x_col
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=float('inf'))
        
        # Compute argmin for this chunk
        for i in range(BLOCK_SIZE):
            if mask[i] and x_vals[i] < local_min_val[i]:
                local_min_val[i] = x_vals[i]
                local_min_idx[i] = col_offset[i]
        
        # Synchronize threads before next iteration
        tl.sync()
    
    # Reduction within block using shared memory
    for i in range(BLOCK_SIZE // 2):
        if i < DIM_SIZE:
            if local_min_val[i] > local_min_val[i + BLOCK_SIZE // 2]:
                local_min_val[i] = local_min_val[i + BLOCK_SIZE // 2]
                local_min_idx[i] = local_min_idx[i + BLOCK_SIZE // 2]
    
    # Store result
    if row < n_elements // DIM_SIZE:
        tl.store(out_ptr + row, local_min_idx[0])

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != 1:
            # Fall back to PyTorch implementation for non-standard dimensions
            return torch.argmin(x, dim=self.dim)
        
        # For dim=1, use Triton kernel
        batch_size, dim1, dim2 = x.shape
        
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Prepare output tensor
        out = torch.empty(batch_size, dtype=torch.int64, device=x.device)
        
        # Kernel configuration
        BLOCK_SIZE = 128
        grid_size = batch_size
        
        # Launch kernel
        argmin_kernel[grid_size](
            x,
            out,
            batch_size * dim2,
            x.stride(0),
            x.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            DIM_SIZE=dim2
        )
        
        return out

# For cases where we can't use the custom kernel, fall back to PyTorch
def get_inputs():
    x = torch.rand(128, 4096, 4095)
    return [x]

def get_init_inputs():
    return [1]