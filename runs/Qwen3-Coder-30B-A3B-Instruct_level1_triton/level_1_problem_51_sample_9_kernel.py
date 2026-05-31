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
    
    # Process the row in chunks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual column indices
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # Load values
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=float('-inf'))
        
        # Find local max
        local_max = tl.max(vals)
        local_max_idx = tl.argmax(vals, axis=0)
        
        # Update global max
        new_max_mask = local_max > max_val
        max_val = tl.where(new_max_mask, local_max, max_val)
        max_idx = tl.where(new_max_mask, local_max_idx + col_start, max_idx)
    
    # Store result
    tl.store(output_ptr + row_idx, max_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 2:
            # For argmax along last dimension, use optimized kernel
            assert len(x.shape) == 3
            batch_size, dim1, dim2 = x.shape
            
            # Ensure input is contiguous and on GPU
            x = x.contiguous().cuda()
            
            # Prepare output
            output = torch.empty(batch_size, dim1, dtype=torch.int32, device=x.device)
            
            # Kernel parameters
            BLOCK_SIZE = 1024
            grid = (batch_size * dim1,)
            
            # Launch kernel
            argmax_kernel[grid](
                x,
                output,
                batch_size,
                dim2,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            return output
        else:
            # Fall back to PyTorch for other dimensions
            return torch.argmax(x, dim=self.dim)