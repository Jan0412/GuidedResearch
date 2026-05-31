import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def scan_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and element indices
    batch_idx = tl.program_id(0)
    # Process one batch at a time
    batch_offset = batch_idx * dim_size
    
    # For each element in the dimension, compute prefix sum
    for i in range(dim_size):
        # Calculate the offset for current element
        current_offset = batch_offset + i
        
        # Load the current value
        x_val = tl.load(x_ptr + current_offset, mask=current_offset < n_elements, other=0.0)
        
        # If not the first element, accumulate from previous
        if i > 0:
            prev_offset = batch_offset + i - 1
            prev_val = tl.load(out_ptr + prev_offset, mask=prev_offset < n_elements, other=0.0)
            x_val = x_val + prev_val
            
        # Store the result
        tl.store(out_ptr + current_offset, x_val, mask=current_offset < n_elements)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Ensure input is contiguous for efficient memory access
        x = x.contiguous()
        
        # Get dimensions
        batch_size = x.shape[0]
        dim_size = x.shape[self.dim] if self.dim < len(x.shape) else 1
        total_elements = x.numel()
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Set block size
        BLOCK_SIZE = 128
        
        # Grid size based on batch size
        grid = (batch_size,)
        
        # Launch kernel
        scan_kernel[grid](
            x,
            out,
            total_elements,
            dim_size,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out