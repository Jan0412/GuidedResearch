import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scan_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate the starting position for this batch and sequence
    start_pos = batch_idx * seq_len + seq_idx * BLOCK_SIZE
    
    # Process elements in chunks of BLOCK_SIZE
    for i in range(0, seq_len, BLOCK_SIZE):
        # Calculate actual offset within sequence
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < seq_len
        
        # Load input data
        x_vals = tl.load(x_ptr + start_pos + offset, mask=mask, other=0.0)
        
        # Compute inclusive prefix sum
        cumsum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(BLOCK_SIZE):
            if offset[j] < seq_len:
                if j == 0:
                    cumsum[j] = x_vals[j]
                else:
                    cumsum[j] = cumsum[j-1] + x_vals[j]
        
        # Store results
        tl.store(y_ptr + start_pos + offset, cumsum, mask=mask)

def triton_scan(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative sum using a two-stage approach:
    1. Process each batch independently
    2. Handle sequential operations within each batch
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Ensure we're working along the correct dimension
    if dim < 0:
        dim += x.dim()
    
    # For simplicity, we'll assume dim=1 as per the problem setup
    # In a full implementation, we'd handle arbitrary dimensions
    batch_size, seq_len = x.shape[0], x.shape[1]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size for efficient memory access
    BLOCK_SIZE = 128
    
    # Grid configuration: one block per batch, one block per sequence
    grid = (batch_size, seq_len // BLOCK_SIZE + (1 if seq_len % BLOCK_SIZE != 0 else 0))
    
    # Launch kernel
    scan_kernel[grid](x, out, x.numel(), batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_scan(x, self.dim)