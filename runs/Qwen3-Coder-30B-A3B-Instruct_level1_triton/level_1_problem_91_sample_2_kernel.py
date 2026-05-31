import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reverse_cumsum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride_inner,
    stride_outer,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_id = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_idx = block_id * BLOCK_SIZE
    
    # Handle the case where we're processing the entire tensor
    if start_idx >= n_elements:
        return
    
    # For each element in the block
    for i in range(BLOCK_SIZE):
        idx = start_idx + i
        if idx >= n_elements:
            break
            
        # Calculate the position in the original tensor
        outer_idx = idx // dim_size
        inner_idx = idx % dim_size
        
        # Calculate cumulative sum from right to left
        cumsum_val = 0.0
        for j in range(dim_size - 1, inner_idx - 1, -1):
            # Calculate actual index in the tensor
            actual_idx = outer_idx * stride_outer + j * stride_inner
            val = tl.load(input_ptr + actual_idx)
            cumsum_val += val
            # Store result at the corresponding position
            output_idx = outer_idx * stride_outer + inner_idx * stride_inner
            tl.store(output_ptr + output_idx, cumsum_val)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # We'll implement a custom Triton kernel that directly computes
        # the reverse cumulative sum in one pass without flipping
        return self._reverse_cumsum_triton(x, self.dim)
    
    def _reverse_cumsum_triton(self, x, dim):
        # Get the shape and strides
        shape = x.shape
        strides = x.stride()
        
        # Calculate total elements
        total_elements = x.numel()
        
        # Create output tensor
        output = torch.empty_like(x)
        
        # Special handling for different dimensions
        if dim == -1:
            dim = len(shape) - 1
            
        # Get the size along the specified dimension
        dim_size = shape[dim]
        if dim_size <= 0:
            return output
            
        # Calculate strides for efficient memory access
        stride_inner = strides[dim]
        stride_outer = 1
        for i in range(len(strides)):
            if i != dim:
                stride_outer *= strides[i]
                
        # Determine grid size based on total elements
        BLOCK_SIZE = 1024
        grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch the kernel
        reverse_cumsum_kernel[grid_size](
            x,
            output,
            total_elements,
            dim_size,
            stride_inner,
            stride_outer,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output