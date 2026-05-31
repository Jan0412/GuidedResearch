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
    # Get the batch and position within the sequence
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate base pointers for this batch and sequence
    batch_offset = batch_idx * dim_size
    seq_offset = seq_idx * dim_size
    
    # Process elements in blocks
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate actual indices
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < dim_size
        
        # Load data
        x_vals = tl.load(x_ptr + batch_offset + seq_offset + idx, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptr + batch_offset + seq_offset + idx, mask=mask, other=False)
        
        # Apply mask and compute cumulative sum
        masked_vals = tl.where(mask_vals, x_vals, 0.0)
        
        # Compute cumulative sum manually since Triton doesn't have cumsum
        cumsum_val = 0.0
        for j in range(BLOCK_SIZE):
            if j < dim_size - i:
                cumsum_val += masked_vals[j]
                tl.store(out_ptr + batch_offset + seq_offset + i + j, cumsum_val, mask=(i + j) < dim_size)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        # For simplicity, we'll handle the case where we do cumulative sum along last dimension
        # This is a simplified version that processes each sequence independently
        batch_size = x.size(0)
        seq_len = x.size(-1)  # Assuming last dimension is the sequence length
        
        # Ensure inputs are contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        
        # Initialize output
        out = torch.empty_like(x)
        
        # Define block size
        BLOCK_SIZE = 128
        
        # Calculate grid dimensions
        grid = (batch_size, seq_len // BLOCK_SIZE + (1 if seq_len % BLOCK_SIZE != 0 else 0))
        
        # Launch kernel
        masked_cumsum_kernel[grid](
            x,
            mask,
            out,
            x.numel(),
            seq_len,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out