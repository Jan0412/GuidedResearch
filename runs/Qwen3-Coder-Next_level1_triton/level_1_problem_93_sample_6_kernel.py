import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    mask_ptr,  # Mask tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length (size along dim)
    stride_batch,  # Stride between batches
    stride_seq,  # Stride along the sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one batch
    batch_idx = tl.program_id(0)
    
    # Calculate base pointers for this batch
    x_offset = batch_idx * stride_batch
    mask_offset = batch_idx * stride_batch
    out_offset = batch_idx * stride_batch
    
    # Initialize cumulative sum to zero
    cumsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process sequence in blocks
    for start in range(0, seq_len, BLOCK_SIZE):
        # Calculate offsets for current block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load input values and mask
        x_val = tl.load(x_ptr + x_offset + offsets * stride_seq, mask=mask, other=0.0)
        mask_val = tl.load(mask_ptr + mask_offset + offsets * stride_seq, mask=mask, other=0)
        
        # Convert mask to float for computation
        mask_float = mask_val.to(tl.float32)
        
        # Update cumulative sum: only add when mask is True
        cumsum = cumsum + x_val * mask_float
        
        # Store the current cumulative sum
        tl.store(out_ptr + out_offset + offsets * stride_seq, cumsum, mask=mask)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Computes masked cumulative sum along specified dimension.
    
    Args:
        x: Input tensor
        mask: Boolean mask of same shape as x
        dim: Dimension along which to compute cumulative sum
        
    Returns:
        Tensor with cumulative sum of x where mask is True
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    assert x.shape == mask.shape, "Input and mask must have same shape."
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    batch_size = 1
    seq_len = x.shape[dim]
    
    # Reshape to 2D if needed for easier processing
    if dim != 1:
        # Reshape to (batch, seq) where batch is all dimensions except dim
        permute_dims = list(range(x.ndim))
        # Move dim to position 1
        permute_dims[1], permute_dims[dim] = permute_dims[dim], permute_dims[1]
        x = x.permute(permute_dims).contiguous()
        mask = mask.permute(permute_dims).contiguous()
        
        # Calculate batch size as product of all dimensions except seq
        batch_size = 1
        for i, s in enumerate(x.shape):
            if i != 1:
                batch_size *= s
        seq_len = x.shape[1]
        
        # Reshape to 2D [batch_size, seq_len]
        x = x.view(batch_size, seq_len)
        mask = mask.view(batch_size, seq_len)
        out = out.view(batch_size, seq_len)
    
    # Get strides
    stride_batch = x.stride(0)
    stride_seq = x.stride(1)
    
    # Set block size
    BLOCK_SIZE = min(1024, seq_len)
    
    # Create grid: one block per batch
    grid = (batch_size,)
    
    # Launch kernel
    masked_cumsum_kernel[grid](
        x, mask, out,
        batch_size, seq_len,
        stride_batch, stride_seq,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back if we modified dimensions
    if dim != 1:
        out = out.view(x.shape)
        # Reverse the permutation
        reverse_permute = [0] * len(permute_dims)
        for i, d in enumerate(permute_dims):
            reverse_permute[d] = i
        out = out.permute(reverse_permute).contiguous()
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using Triton kernel.
    
    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
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