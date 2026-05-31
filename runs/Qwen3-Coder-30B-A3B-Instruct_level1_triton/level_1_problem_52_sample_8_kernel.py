import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def argmin_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_x_batch,
    stride_x_dim,
    stride_x_other,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    batch_start = batch_idx * stride_x_batch
    
    # For each element in the output (which corresponds to one batch)
    for i in range(tl.cdiv(dim_size, BLOCK_SIZE)):
        # Calculate the offset within the current batch
        block_start = i * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask to prevent going out of bounds
        mask = offsets < dim_size
        
        # Load data from memory
        x_data = tl.load(x_ptr + batch_start + offsets * stride_x_dim, mask=mask, other=tl.inf)
        
        # Find the minimum value and its index within this block
        min_val = tl.min(x_data)
        min_idx = tl.argmin(x_data)
        
        # Store the result for this batch
        tl.store(output_ptr + batch_idx, min_idx)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 1:
            # Special handling for the case where we're finding argmin along dimension 1
            # For this specific case, we can optimize using Triton
            batch_size, dim1, dim2 = x.shape
            
            # Allocate output tensor
            output = torch.empty(batch_size, dtype=torch.int64, device=x.device)
            
            # Ensure inputs are contiguous
            x = x.contiguous()
            
            # Set up Triton kernel launch parameters
            BLOCK_SIZE = 1024
            
            # Grid size based on batch size
            grid = (batch_size,)
            
            # Call the Triton kernel
            argmin_kernel[grid](
                x,
                output,
                x.numel(),
                dim1,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            return output
        else:
            # Fall back to PyTorch for other dimensions
            return torch.argmin(x, dim=self.dim)