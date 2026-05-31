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
    # Get the batch and position within the dimension
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1)
    
    # Calculate the starting offset for this batch and dimension
    base_offset = batch_idx * dim_size + dim_idx * dim_size
    
    # Process elements in chunks
    block_start = tl.program_id(2) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim_size
    
    # Load data
    x_vals = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
    mask_vals = tl.load(mask_ptr + base_offset + offsets, mask=mask, other=False)
    
    # Apply mask and compute cumulative sum
    masked_vals = tl.where(mask_vals, x_vals, 0.0)
    
    # Compute cumulative sum manually (forward pass)
    cumsum_val = 0.0
    for i in range(BLOCK_SIZE):
        if offsets[i] < dim_size:
            cumsum_val += masked_vals[i]
            tl.store(out_ptr + base_offset + offsets[i], cumsum_val, mask=mask[i])

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Triton implementation of masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA"
    assert x.shape == mask.shape, "Input and mask must have the same shape"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Prepare output tensor
    out = torch.zeros_like(x)
    
    # Get dimensions
    shape = x.shape
    batch_size = shape[0]
    dim_size = shape[dim]
    
    # For simplicity, we'll handle each batch separately
    # In practice, you'd want more sophisticated indexing
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = (
        batch_size,  # Batch dimension
        1,           # We'll process one dimension at a time
        (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE  # Blocks per dimension
    )
    
    # Launch kernel
    masked_cumsum_kernel[grid](
        x,
        mask,
        out,
        x.numel(),
        dim_size,
        batch_size,
        BLOCK_SIZE=BLOCK_SIZE
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