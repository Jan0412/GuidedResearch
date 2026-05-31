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
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    batch_start = batch_idx * dim_size
    
    # Process elements in chunks
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate global offset for current chunk
        block_start = batch_start + i
        
        # Create offsets within the block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid elements
        mask = offsets < batch_start + dim_size
        
        # Load x values
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Load mask values
        mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=False)
        
        # Apply mask to x values
        masked_x = tl.where(mask_vals, x_vals, 0.0)
        
        # Compute cumulative sum within the block
        cumsum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(BLOCK_SIZE):
            if i + j < dim_size:
                cumsum[j] = tl.sum(masked_x[:j+1]) if j == 0 else cumsum[j-1] + masked_x[j]
        
        # Store results
        tl.store(out_ptr + offsets, cumsum, mask=mask)

# Optimized version using a more efficient approach
@triton.jit
def masked_cumsum_kernel_optimized(
    x_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    dim_size,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    batch_start = batch_idx * dim_size
    
    # Process in blocks
    for i in range(0, dim_size, BLOCK_SIZE):
        block_start = batch_start + i
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < batch_start + dim_size
        
        # Load data
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=False)
        
        # Apply mask
        masked_x = tl.where(mask_vals, x_vals, 0.0)
        
        # Compute cumulative sum manually
        cumsum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(BLOCK_SIZE):
            if i + j < dim_size:
                if j == 0:
                    cumsum[j] = masked_x[j]
                else:
                    cumsum[j] = cumsum[j-1] + masked_x[j]
        
        # Store results
        tl.store(out_ptr + offsets, cumsum, mask=mask)

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Triton implementation of masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA"
    assert x.shape == mask.shape, "x and mask must have the same shape"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate dimensions
    batch_size = x.shape[0]
    dim_size = x.shape[dim]
    n_elements = x.numel()
    
    # Choose block size
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = (batch_size,)
    
    # Launch kernel
    masked_cumsum_kernel_optimized[grid](
        x, mask, out, n_elements, dim_size, batch_size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    Optimized with custom Triton kernels.
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