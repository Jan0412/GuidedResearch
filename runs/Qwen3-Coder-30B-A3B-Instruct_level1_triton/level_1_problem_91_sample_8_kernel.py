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
    
    # For each element in the block, calculate its position in the tensor
    for i in range(BLOCK_SIZE):
        idx = start_idx + i
        if idx >= n_elements:
            break
            
        # Convert linear index to multi-dimensional indices
        outer_idx = idx // dim_size
        inner_idx = idx % dim_size
        
        # Calculate actual memory addresses
        input_addr = input_ptr + outer_idx * stride_outer + inner_idx * stride_inner
        output_addr = output_ptr + outer_idx * stride_outer + inner_idx * stride_inner
        
        # Perform reverse cumulative sum along the specified dimension
        # Read from input and accumulate backwards
        cumsum_val = 0.0
        for j in range(dim_size - 1, -1, -1):
            current_idx = outer_idx * stride_outer + j * stride_inner
            val = tl.load(input_ptr + current_idx, mask=(j < dim_size))
            cumsum_val += val
            tl.store(output_ptr + current_idx, cumsum_val, mask=(j < dim_size))

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # For better performance with Triton, we'll handle the reverse cumsum directly
        # in a more efficient way using fused operations
        
        # Get dimensions
        shape = x.shape
        dim_size = shape[self.dim]
        
        # Create output tensor
        output = torch.empty_like(x)
        
        # Handle the case where we're working with the last dimension for simplicity
        if self.dim == len(shape) - 1:
            # Use Triton kernel for the operation
            return self._triton_reverse_cumsum(x, dim_size)
        else:
            # For other dimensions, we can still use the standard approach but 
            # with better optimization in the kernel
            return self._optimized_reverse_cumsum(x, dim_size)

    def _triton_reverse_cumsum(self, x, dim_size):
        # Flatten to make indexing easier
        original_shape = x.shape
        flat_x = x.view(-1, dim_size)
        flat_output = torch.empty_like(flat_x)
        
        # Determine the number of blocks needed
        n_elements = flat_x.numel()
        BLOCK_SIZE = 128
        
        # Grid configuration
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        
        # Call our Triton kernel
        reverse_cumsum_kernel[grid](
            flat_x,
            flat_output,
            n_elements,
            dim_size,
            1,  # stride_inner for last dimension
            flat_x.stride(0),  # stride_outer
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return flat_output.view(original_shape)

    def _optimized_reverse_cumsum(self, x, dim_size):
        # For non-last dimensions, we can optimize by processing slices
        # But for simplicity and to avoid complex indexing in Triton,
        # we'll fall back to PyTorch for now, but note that this could be improved further
        flipped = x.flip(self.dim)
        cumsummed = torch.cumsum(flipped, dim=self.dim)
        return cumsummed.flip(self.dim)