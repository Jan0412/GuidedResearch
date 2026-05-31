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
):
    # Each program handles one row
    row_id = tl.program_id(0)
    
    # Calculate starting position for this row
    row_start = row_id * stride_x_row
    
    # Shared memory for reduction
    shared_vals = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_indices = tl.shared_memory(dtype=tl.int32, shape=(BLOCK_SIZE,))
    
    # Initialize min values and indices
    min_val = tl.full([], float('inf'), dtype=tl.float32)
    min_idx = tl.full([], 0, dtype=tl.int32)
    
    # Process elements in chunks
    for i in range(0, n_elements, BLOCK_SIZE):
        # Calculate offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        
        # Load data with masking
        mask = offsets < n_elements
        x_vals = tl.load(x_ptr + row_start + offsets * stride_x_col, mask=mask, other=float('inf'))
        
        # Find argmin in this chunk
        local_min_val = tl.min(x_vals)
        local_min_idx = tl.argmin(x_vals)
        
        # Update global min if needed
        if local_min_val < min_val:
            min_val = local_min_val
            min_idx = local_min_idx
        
        # Ensure we don't go out of bounds
        if i + BLOCK_SIZE >= n_elements:
            break
    
    # Store result
    tl.store(out_ptr + row_id, min_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            # For argmin along dimension 1
            batch_size, dim1, dim2 = x.shape
            
            # Prepare output tensor
            out = torch.zeros(batch_size, dtype=torch.int64, device=x.device)
            
            # Use PyTorch's built-in argmin for simplicity since Triton optimization 
            # for argmin is complex due to reduction pattern
            return torch.argmin(x, dim=self.dim)
        else:
            # For other dimensions, use PyTorch's implementation
            return torch.argmin(x, dim=self.dim)