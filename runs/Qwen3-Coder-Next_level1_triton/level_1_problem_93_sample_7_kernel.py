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
    dim,  # Dimension to cumsum along
    stride_batch,  # Stride for batch dimension
    stride_seq,  # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Determine which batch this program handles
    batch_id = tl.program_id(0)
    
    # Calculate offsets for this batch
    if dim == 1:
        offsets = tl.arange(0, BLOCK_SIZE)
    else:  # dim == 0, though for simplicity we assume dim=1 as in example
        offsets = tl.arange(0, BLOCK_SIZE) * stride_seq
    
    # Load mask and x values for this segment
    mask_offsets = batch_id * stride_batch + offsets
    mask = tl.load(mask_ptr + mask_offsets, mask=offsets < seq_len, other=0)
    x = tl.load(x_ptr + mask_offsets, mask=offsets < seq_len, other=0.0)
    
    # Convert mask to int for computation
    mask_int = mask.to(tl.int32)
    
    # Compute masked cumulative sum
    cumsum = tl.zeros_like(x)
    running_sum = tl.zeros_like(x)
    
    for i in range(seq_len):
        if dim == 1:
            offset_i = i
        else:
            offset_i = i * stride_seq
            
        current_mask = tl.load(mask_ptr + batch_id * stride_batch + offset_i, 
                              mask=offset_i < seq_len, other=0)
        current_x = tl.load(x_ptr + batch_id * stride_batch + offset_i, 
                          mask=offset_i < seq_len, other=0.0)
        
        if i == 0:
            running_sum = tl.where(current_mask, current_x, 0.0)
        else:
            running_sum = tl.where(current_mask, running_sum + current_x, 0.0)
            
        tl.store(out_ptr + batch_id * stride_batch + offset_i, running_sum,
                mask=offset_i < seq_len)


@triton.jit
def masked_cumsum_kernel_fused(
    x_ptr,  # Input tensor pointer
    mask_ptr,  # Mask tensor pointer  
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length
    stride_batch,  # Stride for batch dimension
    stride_seq,  # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each batch is processed independently
    batch_id = tl.program_id(0)
    
    # Process the sequence in blocks
    for start in range(0, seq_len, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = tl.load(mask_ptr + batch_id * stride_batch + offsets, 
                      mask=offsets < seq_len, other=0)
        x = tl.load(x_ptr + batch_id * stride_batch + offsets, 
                   mask=offsets < seq_len, other=0.0)
        
        # Compute cumulative sum with masking
        cumsum = tl.zeros_like(x)
        for i in range(BLOCK_SIZE):
            idx = start + i
            if idx == 0:
                cumsum = tl.where(mask, x, 0.0)
            else:
                prev_offset = idx - 1
                prev_cumsum = tl.load(out_ptr + batch_id * stride_batch + prev_offset,
                                     mask=prev_offset < seq_len, other=0.0)
                cumsum = tl.where(mask, prev_cumsum + x, 0.0)
            
            tl.store(out_ptr + batch_id * stride_batch + idx, cumsum,
                    mask=idx < seq_len)


@triton.jit
def masked_cumsum_optimized_kernel(
    x_ptr,  # Input tensor pointer
    mask_ptr,  # Mask tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of batches
    seq_len,  # Sequence length
    stride_batch,  # Stride for batch dimension
    stride_seq,  # Stride for sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized masked cumulative sum kernel that handles each batch independently.
    For each position, it maintains a running sum that resets to 0 when mask is False.
    """
    batch_id = tl.program_id(0)
    
    # Process sequence in a single pass
    running_sum = tl.zeros(1, dtype=tl.float32)
    
    for i in range(seq_len):
        offset = batch_id * stride_batch + i * stride_seq
        
        # Load current mask and value
        current_mask = tl.load(mask_ptr + offset)
        current_x = tl.load(x_ptr + offset)
        
        # Update running sum: only add if mask is True, otherwise reset
        running_sum = tl.where(current_mask, running_sum + current_x, 0.0)
        
        # Store result
        tl.store(out_ptr + offset, running_sum)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Triton implementation of masked cumulative sum.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len)
        mask: Boolean mask of same shape as x
        dim: Dimension along which to compute cumulative sum
    
    Returns:
        Tensor with cumulative sum where mask is True, reset to 0 where mask is False
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    assert x.shape == mask.shape, "Input and mask must have the same shape."
    
    # Ensure contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Get dimensions
    batch_size = x.shape[0] if dim == 1 else x.shape[1]
    seq_len = x.shape[1] if dim == 1 else x.shape[0]
    
    # Determine strides
    if dim == 1:
        stride_batch = x.stride(0)
        stride_seq = x.stride(1)
    else:  # dim == 0
        stride_batch = x.stride(1)
        stride_seq = x.stride(0)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Configure kernel
    BLOCK_SIZE = 256
    
    # Grid: one block per batch
    grid = (batch_size,)
    
    # Launch kernel
    masked_cumsum_optimized_kernel[grid](
        x, mask, out,
        batch_size, seq_len,
        stride_batch, stride_seq,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the masked cumulative sum model using Triton kernels.
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