import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cumulative_product_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    batch_size,
    stride_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_idx = block_id * BLOCK_SIZE
    
    # For each element in the block, compute cumulative product along the specified dimension
    for i in range(start_idx, min(start_idx + BLOCK_SIZE, n_elements)):
        # Calculate batch and position within dimension
        batch_idx = i // (dim_size * stride_dim)
        pos_in_dim = (i % (dim_size * stride_dim)) // stride_dim
        
        # Compute cumulative product up to current position
        if pos_in_dim == 0:
            # First element, just copy it
            tl.store(output_ptr + i, tl.load(input_ptr + i))
        else:
            # Compute cumulative product from previous element
            prev_idx = i - stride_dim
            curr_val = tl.load(input_ptr + i)
            prev_val = tl.load(output_ptr + prev_idx)
            tl.store(output_ptr + i, curr_val * prev_val)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get dimensions
        shape = x.shape
        dim_size = shape[self.dim]
        batch_size = 1
        for i in range(len(shape)):
            if i != self.dim:
                batch_size *= shape[i]
        
        # Calculate strides
        stride_dim = 1
        for i in range(self.dim + 1, len(shape)):
            stride_dim *= shape[i]
            
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Total number of elements
        n_elements = x.numel()
        
        # Set block size
        BLOCK_SIZE = 1024
        
        # Grid size
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        
        # Launch kernel
        cumulative_product_kernel[grid](
            x, 
            output, 
            n_elements, 
            dim_size, 
            batch_size, 
            stride_dim, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output