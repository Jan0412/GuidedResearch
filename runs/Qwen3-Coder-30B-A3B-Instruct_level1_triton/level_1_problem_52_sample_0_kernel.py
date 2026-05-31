import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmin_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_x_row,
    stride_x_col,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program processes one row
    if row_idx >= n_elements // dim_size:
        return
    
    # Calculate starting offset for this row
    row_start = row_idx * stride_x_row
    
    # Initialize min_val and min_idx
    min_val = tl.full([1], float('inf'), dtype=tl.float32)
    min_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process elements in this row
    for col in range(0, dim_size, BLOCK_SIZE):
        # Calculate offsets for current block
        offsets = col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load values from memory
        x_vals = tl.load(x_ptr + row_start + offsets * stride_x_col, mask=mask, other=float('inf'))
        
        # Find min in this block
        block_min_val = tl.min(x_vals)
        block_min_idx = tl.argmin(x_vals)
        
        # Update global min
        new_min_mask = block_min_val < min_val
        min_val = tl.where(new_min_mask, block_min_val, min_val)
        min_idx = tl.where(new_min_mask, block_min_idx + col, min_idx)
    
    # Store result
    tl.store(output_ptr + row_idx, min_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            # For dim=1 case, we need to find argmin along that dimension
            batch_size, dim1, dim2 = x.shape
            
            # Create output tensor
            output = torch.empty(batch_size, dim1, dtype=torch.int64, device=x.device)
            
            # Ensure input is contiguous
            x = x.contiguous()
            
            # Calculate grid size
            grid_size = batch_size * dim1
            
            # Launch kernel
            argmin_kernel[(grid_size,)](
                x,
                output,
                batch_size * dim1 * dim2,
                dim2,
                x.stride(0),
                x.stride(1),
                BLOCK_SIZE=128
            )
            
            return output
        else:
            # For other dimensions, fall back to PyTorch implementation
            return torch.argmin(x, dim=self.dim)