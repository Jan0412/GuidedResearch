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
    x_base = batch_idx * dim_size + seq_idx * dim_size
    mask_base = batch_idx * dim_size + seq_idx * dim_size
    out_base = batch_idx * dim_size + seq_idx * dim_size
    
    # Process elements in blocks
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate current offset and limit
        offset = i
        limit = min(i + BLOCK_SIZE, dim_size)
        
        # Load data for this block
        x_offsets = x_base + tl.arange(0, BLOCK_SIZE)
        mask_offsets = mask_base + tl.arange(0, BLOCK_SIZE)
        out_offsets = out_base + tl.arange(0, BLOCK_SIZE)
        
        # Create masks for valid elements
        x_mask = offset + tl.arange(0, BLOCK_SIZE) < dim_size
        mask_mask = offset + tl.arange(0, BLOCK_SIZE) < dim_size
        
        # Load values
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        mask_vals = tl.load(mask_ptr + mask_offsets, mask=mask_mask, other=False)
        
        # Apply mask and compute cumulative sum
        masked_vals = tl.where(mask_vals, x_vals, 0.0)
        
        # Compute cumulative sum within block
        cumsum_val = 0.0
        for j in range(limit - offset):
            cumsum_val += masked_vals[j]
            tl.store(out_ptr + out_base + offset + j, cumsum_val, mask=(offset + j) < dim_size)

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
    out = torch.empty_like(x)
    
    # Handle different dimensions
    if dim == 1:
        batch_size, seq_len = x.shape
        BLOCK_SIZE = 128
        
        # Grid configuration
        grid = (
            batch_size,  # Batch dimension
            1            # Sequence dimension (processed in one block per sequence)
        )
        
        # Launch kernel
        masked_cumsum_kernel[grid](
            x,
            mask,
            out,
            x.numel(),
            seq_len,
            batch_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # For other dimensions, fall back to PyTorch
        return torch.cumsum(x * mask, dim=dim)
    
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