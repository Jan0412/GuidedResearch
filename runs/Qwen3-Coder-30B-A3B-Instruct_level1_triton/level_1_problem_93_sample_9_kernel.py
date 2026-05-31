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
    
    # Calculate the starting offset for this batch and sequence
    base_offset = batch_idx * dim_size + seq_idx * dim_size
    
    # Process elements in chunks
    for i in range(0, dim_size, BLOCK_SIZE):
        # Calculate actual position in the flattened tensor
        pos = base_offset + i + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid positions
        mask = pos < base_offset + dim_size
        
        # Load data
        x_vals = tl.load(x_ptr + pos, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptr + pos, mask=mask, other=False)
        
        # Apply mask and compute cumulative sum
        masked_vals = tl.where(mask_vals, x_vals, 0.0)
        
        # Compute cumulative sum manually since Triton doesn't have cumsum
        # We'll use a simple loop approach for small blocks
        cumsum_val = 0.0
        for j in range(BLOCK_SIZE):
            if i + j < dim_size:
                cumsum_val += masked_vals[j]
                tl.store(out_ptr + pos[j], cumsum_val, mask=pos[j] < base_offset + dim_size)

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Triton implementation of masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA"
    assert x.shape == mask.shape, "x and mask must have the same shape"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate dimensions
    batch_size = x.shape[0]
    dim_size = x.shape[dim]
    
    # For simplicity, we'll handle the case where we process each sequence separately
    # This is more efficient for our kernel design
    BLOCK_SIZE = 128
    
    # Grid configuration: one block per batch element
    grid = (batch_size, dim_size // BLOCK_SIZE + 1)
    
    # Launch kernel
    masked_cumsum_kernel[grid](
        x, mask, out, 
        x.numel(), dim_size, batch_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Since the above approach has issues with proper cumulative sum tracking across blocks,
# let's rewrite it with a better approach that handles the cumulative nature correctly

@triton.jit
def masked_cumsum_kernel_v2(
    x_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sequence (batch element)
    batch_idx = tl.program_id(0)
    
    # Base offset for this batch element
    base_offset = batch_idx * seq_len
    
    # Initialize cumulative sum
    cumsum_val = 0.0
    
    # Process elements sequentially
    for i in range(seq_len):
        # Calculate position
        pos = base_offset + i
        
        # Load data
        x_val = tl.load(x_ptr + pos)
        mask_val = tl.load(mask_ptr + pos)
        
        # Apply mask
        masked_val = tl.where(mask_val, x_val, 0.0)
        
        # Update cumulative sum
        cumsum_val += masked_val
        
        # Store result
        tl.store(out_ptr + pos, cumsum_val)

def triton_masked_cumsum_v2(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Improved Triton implementation of masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA"
    assert x.shape == mask.shape, "x and mask must have the same shape"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate dimensions
    batch_size = x.shape[0]
    seq_len = x.shape[dim] if dim >= 0 else x.shape[-1]
    
    # Grid configuration: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    masked_cumsum_kernel_v2[grid](
        x, mask, out,
        batch_size, seq_len,
        BLOCK_SIZE=128
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized version using Triton kernels for masked cumulative sum
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
        return triton_masked_cumsum_v2(x, mask, self.dim)