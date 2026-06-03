import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_elements,  # Total number of elements
    stride,  # Stride along the dimension we're cumsumming
    dim_size,  # Size of the dimension we're cumsumming
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    # Calculate which batch we're processing
    batch_id = tl.program_id(0)
    
    # Calculate base offset for this batch
    base_offset = batch_id * stride * dim_size
    
    # For each position in the dimension we're cumsumming
    dim_offset = tl.program_id(1) * BLOCK_DIM
    
    # Calculate starting position (from the end for reverse cumsum)
    start_pos = dim_size - 1 - dim_offset
    
    # Check if we're within bounds
    if start_pos < 0:
        return
    
    # Calculate offsets for this block
    offsets = base_offset + start_pos * stride + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_id + 1) * stride * dim_size
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Initialize running sum
    cumsum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Perform reverse cumulative sum
    for i in range(min(BLOCK_DIM, dim_size - dim_offset)):
        pos = start_pos - i
        # Calculate actual offsets for this position
        pos_offsets = base_offset + pos * stride + tl.arange(0, BLOCK_SIZE)
        pos_mask = pos_offsets < (batch_id + 1) * stride * dim_size
        
        # Load current value
        val = tl.load(x_ptr + pos_offsets, mask=pos_mask, other=0.0)
        
        # Update running sum
        cumsum = cumsum + val.to(tl.float32)
        
        # Store result
        tl.store(out_ptr + pos_offsets, cumsum.to(x_ptr.dtype.element_ty), mask=pos_mask)


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Compute reverse cumulative sum along specified dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute reverse cumsum
        
    Returns:
        Tensor with reverse cumulative sum computed
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and calculate strides
    shape = x.shape
    stride = x.stride(dim)
    dim_size = shape[dim]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine block sizes
    # Use a reasonable block size for the non-dim dimensions
    non_dim_size = x.numel() // dim_size
    BLOCK_SIZE = min(128, non_dim_size)
    BLOCK_DIM = 16  # Process multiple positions along the dimension
    
    # Calculate grid dimensions
    # One block per batch (all dimensions except dim)
    batch_dims = list(range(len(shape)))
    batch_dims.remove(dim)
    
    if len(batch_dims) == 0:
        # Special case: dim is the only dimension
        num_batches = 1
    else:
        num_batches = 1
        for d in batch_dims:
            num_batches *= shape[d]
    
    num_dim_blocks = (dim_size + BLOCK_DIM - 1) // BLOCK_DIM
    
    grid = (num_batches, num_dim_blocks)
    
    # Launch kernel
    reverse_cumsum_kernel[grid](
        x, out, x.numel(), stride, dim_size,
        BLOCK_SIZE=BLOCK_SIZE, BLOCK_DIM=BLOCK_DIM
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation along 
    a specified dimension using custom Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)