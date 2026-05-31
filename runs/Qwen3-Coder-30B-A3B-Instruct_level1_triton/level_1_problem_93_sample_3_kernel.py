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
    
    # Each program handles one element in the sequence
    block_start = tl.program_id(2) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim_size
    
    # Load data
    x_vals = tl.load(x_ptr + batch_offset + seq_offset + offsets, mask=mask, other=0.0)
    mask_vals = tl.load(mask_ptr + batch_offset + seq_offset + offsets, mask=mask, other=False)
    
    # Apply mask and compute cumulative sum
    masked_x = tl.where(mask_vals, x_vals, 0.0)
    
    # Compute cumulative sum manually (simplified approach)
    cumsum_val = 0.0
    for i in range(dim_size):
        if i >= block_start and i < block_start + BLOCK_SIZE:
            cumsum_val += masked_x[i - block_start] if mask[i - block_start] else 0.0
            tl.store(out_ptr + batch_offset + seq_offset + i, cumsum_val)
    
    # For better performance, we'll compute it in a more efficient way using shared memory
    # But for simplicity and correctness in this case, we'll do sequential accumulation
    # In practice, you'd want to use proper fused kernels with shared memory


@triton.jit
def masked_cumsum_fused_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (batch_size, seq_len)
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate base offset for this batch
    batch_offset = batch_idx * seq_len
    
    # Shared memory for partial sums
    shared_sum = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Process elements in chunks
    for chunk_start in range(0, seq_len, BLOCK_SIZE):
        # Load elements from global memory
        chunk_offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        mask_chunk = tl.load(mask_ptr + batch_offset + chunk_offsets, mask=chunk_offsets < seq_len, other=False)
        x_chunk = tl.load(x_ptr + batch_offset + chunk_offsets, mask=chunk_offsets < seq_len, other=0.0)
        
        # Apply mask and compute cumulative sum
        masked_x = tl.where(mask_chunk, x_chunk, 0.0)
        
        # Compute prefix sum within the block
        cumsum = 0.0
        for i in range(BLOCK_SIZE):
            if chunk_start + i < seq_len:
                cumsum += masked_x[i]
                tl.store(out_ptr + batch_offset + chunk_start + i, cumsum)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Custom Triton implementation of masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA"
    assert x.shape == mask.shape, "Input and mask must have the same shape"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    batch_size = x.size(0)
    seq_len = x.size(dim)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = (batch_size, seq_len, 1)
    
    # Launch kernel
    # Note: This is a simplified version; a full optimized version would use shared memory
    # and more sophisticated parallel prefix sum algorithms
    masked_cumsum_fused_kernel[grid](x, mask, out, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


# More efficient implementation using proper parallel prefix sum
@triton.jit
def efficient_masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    batch_offset = batch_idx * seq_len
    
    # Shared memory for block-level prefix sums
    shared_data = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_mask = tl.shared_memory(dtype=tl.int32, shape=(BLOCK_SIZE,))
    
    # Process each element
    for i in range(seq_len):
        # Load element
        idx = batch_offset + i
        x_val = tl.load(x_ptr + idx, mask=i < seq_len, other=0.0)
        mask_val = tl.load(mask_ptr + idx, mask=i < seq_len, other=False)
        
        # Apply mask
        masked_val = tl.where(mask_val, x_val, 0.0)
        
        # Simple approach: accumulate sequentially (this is not optimal but correct)
        # For production use, you'd implement proper parallel prefix sum
        if i == 0:
            cumsum = masked_val
        else:
            prev_cumsum = tl.load(out_ptr + batch_offset + i - 1, mask=(i - 1) < seq_len, other=0.0)
            cumsum = prev_cumsum + masked_val
        
        tl.store(out_ptr + idx, cumsum, mask=i < seq_len)


# Even better approach - use a simpler fused operation
@triton.jit
def simple_masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one batch
    batch_idx = tl.program_id(0)
    batch_offset = batch_idx * seq_len
    
    # Sequential processing for correctness
    cumsum = 0.0
    for i in range(seq_len):
        # Load data
        x_val = tl.load(x_ptr + batch_offset + i, mask=i < seq_len, other=0.0)
        mask_val = tl.load(mask_ptr + batch_offset + i, mask=i < seq_len, other=False)
        
        # Apply mask and accumulate
        masked_val = tl.where(mask_val, x_val, 0.0)
        cumsum += masked_val
        
        # Store result
        tl.store(out_ptr + batch_offset + i, cumsum, mask=i < seq_len)


def triton_simple_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Simple but correct Triton implementation of masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA"
    assert x.shape == mask.shape, "Input and mask must have the same shape"
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    batch_size = x.size(0)
    seq_len = x.size(dim)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid configuration: one block per batch
    grid = (batch_size, 1, 1)
    
    # Launch kernel
    simple_masked_cumsum_kernel[grid](
        x, mask, out, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
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
        return triton_simple_masked_cumsum(x, mask, self.dim)