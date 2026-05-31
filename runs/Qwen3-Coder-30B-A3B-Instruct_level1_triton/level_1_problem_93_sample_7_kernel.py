import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    dim_size,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    batch_start = batch_idx * dim_size
    
    # Process each element in the current batch
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate global offset
        global_offset = batch_start + i
        
        # Create offsets within the block
        offsets = global_offset + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid elements
        mask = offsets < batch_start + dim_size
        
        # Load inputs
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=False)
        
        # Apply mask and compute cumulative sum
        masked_x = tl.where(mask_vals, x_vals, 0.0)
        
        # Manual cumulative sum within block
        cumsum_val = 0.0
        for j in range(BLOCK_SIZE):
            if i + j < dim_size:
                offset = batch_start + i + j
                cumsum_val += tl.load(x_ptr + offset, mask=offset < batch_start + dim_size, other=0.0) * \
                             tl.load(mask_ptr + offset, mask=offset < batch_start + dim_size, other=False)
                tl.store(out_ptr + offset, cumsum_val, mask=offset < batch_start + dim_size)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        # Ensure inputs are contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        batch_size = x.shape[0]
        dim_size = x.shape[self.dim]
        n_elements = x.numel()
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Grid configuration
        grid = (batch_size,)
        
        # Launch kernel
        masked_cumsum_kernel[grid](
            x, 
            mask, 
            out,
            n_elements,
            dim_size,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out