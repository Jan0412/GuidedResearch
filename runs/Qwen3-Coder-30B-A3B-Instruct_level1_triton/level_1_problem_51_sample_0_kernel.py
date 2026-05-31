import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # For each element, find the argmax along the specified dimension
    # Since we're doing argmax over a specific dimension, we need to process
    # elements in chunks corresponding to that dimension
    # This implementation assumes we're processing along the last dimension
    
    # Find max value in each row
    row_max = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    row_indices = tl.full([BLOCK_SIZE], 0, dtype=tl.int32)
    
    # Process elements in groups of dim_size (assuming we're working with the last dimension)
    for i in range(dim_size):
        idx = offsets * dim_size + i
        mask_i = idx < n_elements
        val = tl.load(x_ptr + idx, mask=mask_i, other=-float('inf'))
        
        # Update max and indices
        mask_greater = val > row_max
        row_max = tl.where(mask_greater, val, row_max)
        row_indices = tl.where(mask_greater, i, row_indices)
    
    # Store results
    tl.store(output_ptr + offsets, row_indices.to(tl.int32), mask=mask)

# Simplified approach - for argmax over a single dimension, we can do this more efficiently
@triton.jit
def argmax_kernel_simple(
    x_ptr,
    output_ptr,
    batch_size,
    dim1,
    dim2,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block processes one element from the output tensor
    pid = tl.program_id(0)
    
    # Calculate which batch and which position in the remaining dimensions
    if dim2 == 1:
        # If dim2 is 1, we're taking argmax over dim1
        batch_idx = pid // dim1
        pos_idx = pid % dim1
        if batch_idx < batch_size:
            # Find argmax in this slice
            max_val = -float('inf')
            max_idx = 0
            for i in range(dim2):
                val = tl.load(x_ptr + batch_idx * dim1 * dim2 + pos_idx * dim2 + i)
                if val > max_val:
                    max_val = val
                    max_idx = i
            tl.store(output_ptr + pid, max_idx)
    else:
        # More general case - argmax over the last dimension
        batch_idx = pid // (dim1 * dim2)
        rest = pid % (dim1 * dim2)
        pos_idx = rest // dim2
        elem_idx = rest % dim2
        
        if batch_idx < batch_size:
            # Find argmax in this slice
            max_val = -float('inf')
            max_idx = 0
            for i in range(dim2):
                val = tl.load(x_ptr + batch_idx * dim1 * dim2 + pos_idx * dim2 + i)
                if val > max_val:
                    max_val = val
                    max_idx = i
            tl.store(output_ptr + pid, max_idx)

def triton_argmax(x: torch.Tensor, dim: int):
    """
    Triton-based argmax implementation
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # For the given problem: x has shape (batch_size, dim1, dim2)
    # We want to apply argmax along the specified dimension
    batch_size, dim1, dim2 = x.shape
    
    if dim == 0:
        # Argmax over batch dimension - not typically useful for this case
        raise NotImplementedError("Argmax over batch dimension not implemented")
    elif dim == 1:
        # Argmax over dim1 dimension
        output_shape = (batch_size, dim2)
        out = torch.empty(output_shape, dtype=torch.int64, device=x.device)
        n_elements = batch_size * dim2
    elif dim == 2:
        # Argmax over dim2 dimension  
        output_shape = (batch_size, dim1)
        out = torch.empty(output_shape, dtype=torch.int64, device=x.device)
        n_elements = batch_size * dim1
    else:
        raise ValueError(f"Unsupported dimension {dim}")
    
    # Use the simple kernel approach
    BLOCK_SIZE = 1024
    
    # Determine grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    argmax_kernel_simple[grid](
        x, out, batch_size, dim1, dim2, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)