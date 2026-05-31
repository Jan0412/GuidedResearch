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
    
    # Calculate the starting column for this threadblock
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # Shared memory for reduction
    shared_max = tl.shared_tensor(tl.float32, BLOCK_SIZE)
    shared_sum = tl.shared_tensor(tl.float32, BLOCK_SIZE)
    
    # Load data from global memory
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input values
    inp_vals = tl.load(inp_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max value for numerical stability
    local_max = tl.max(inp_vals, axis=0)
    shared_max[tl.arange(0, BLOCK_SIZE)] = local_max
    tl.sync()
    max_val = tl.max(shared_max, axis=0)
    
    # Compute exp(x - max)
    exp_vals = tl.exp(inp_vals - max_val)
    
    # Compute sum of exponentials
    local_sum = tl.sum(exp_vals, axis=0)
    shared_sum[tl.arange(0, BLOCK_SIZE)] = local_sum
    tl.sync()
    sum_val = tl.sum(shared_sum, axis=0)
    
    # Compute softmax
    out_vals = exp_vals / sum_val
    
    # Write results back to global memory
    tl.store(out_ptr + row_idx * n_cols + col_offsets, out_vals, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous and on GPU
        x = x.contiguous().cuda()
        
        batch_size, num_features = x.shape
        
        # Allocate output tensor
        output = torch.empty_like(x)
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid dimensions
        grid_x = batch_size
        grid_y = (num_features + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (grid_x, grid_y)
        
        # Launch kernel
        softmax_kernel[grid](
            x,
            output,
            num_features,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output