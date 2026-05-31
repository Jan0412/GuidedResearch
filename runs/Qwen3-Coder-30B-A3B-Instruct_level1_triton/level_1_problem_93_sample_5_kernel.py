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
    
    # Process elements in chunks
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate global indices
        global_idx = batch_offset + seq_offset + i + tl.arange(0, BLOCK_SIZE)
        mask_idx = global_idx
        
        # Create mask for valid elements
        mask = global_idx < n_elements
        
        # Load inputs
        x_vals = tl.load(x_ptr + mask_idx, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptr + mask_idx, mask=mask, other=False)
        
        # Apply mask
        masked_x = tl.where(mask_vals, x_vals, 0.0)
        
        # Compute cumulative sum (in-order processing)
        cumsum_val = 0.0
        for j in range(BLOCK_SIZE):
            if i + j < dim_size:
                cumsum_val += masked_x[j]
                tl.store(out_ptr + batch_offset + seq_offset + i + j, cumsum_val, mask=mask & (i + j < dim_size))

# Optimized version using shared memory for better performance
@triton.jit
def masked_cumsum_optimized_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    dim_size,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Shared memory for accumulating values
    shared_data = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    batch_offset = batch_idx * dim_size
    seq_offset = seq_idx * dim_size
    
    # Initialize cumulative sum
    cumsum_val = 0.0
    
    # Process elements in chunks
    for chunk_start in range(0, dim_size, BLOCK_SIZE):
        # Load data into shared memory
        for i in range(BLOCK_SIZE):
            idx = chunk_start + i
            if idx < dim_size:
                global_idx = batch_offset + seq_offset + idx
                x_val = tl.load(x_ptr + global_idx, mask=True, other=0.0)
                mask_val = tl.load(mask_ptr + global_idx, mask=True, other=False)
                shared_data[i] = tl.where(mask_val, x_val, 0.0)
            else:
                shared_data[i] = 0.0
        
        # Compute prefix sum in shared memory
        for i in range(1, BLOCK_SIZE):
            if chunk_start + i < dim_size:
                shared_data[i] += shared_data[i-1]
        
        # Write back to global memory
        for i in range(BLOCK_SIZE):
            idx = chunk_start + i
            if idx < dim_size:
                global_idx = batch_offset + seq_offset + idx
                tl.store(out_ptr + global_idx, shared_data[i], mask=True)

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Triton implementation of masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA"
    assert x.shape == mask.shape, "Input and mask shapes must match"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate dimensions
    batch_size = x.shape[0]
    dim_size = x.shape[dim]
    n_elements = x.numel()
    
    # Grid configuration
    grid = (batch_size, dim_size)
    BLOCK_SIZE = 128
    
    # Launch kernel
    masked_cumsum_optimized_kernel[grid](
        x, mask, out, n_elements, dim_size, batch_size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    Optimized with Triton kernels for better performance.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return triton_masked_cumsum(x, mask, self.dim)