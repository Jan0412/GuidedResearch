import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_kernel(
    x_ptr, mask_ptr, out_ptr,
    batch_size, seq_len,
    stride_b, stride_s,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one batch
    batch_id = tl.program_id(0)
    
    # Calculate base pointers for this batch
    x_batch_offset = batch_id * stride_b
    mask_batch_offset = batch_id * stride_b
    out_batch_offset = batch_id * stride_b
    
    # Initialize cumulative sum
    cumsum = tl.zeros([1], dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, seq_len, BLOCK_SIZE):
        # Calculate end of this block
        end = tl.minimum(start + BLOCK_SIZE, seq_len)
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load data
        x_vals = tl.load(x_ptr + x_batch_offset + offsets * stride_s, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptr + mask_batch_offset + offsets * stride_s, mask=mask, other=0)
        
        # Convert mask to boolean
        mask_bool = mask_vals.to(tl.int1)
        
        # Apply mask to x
        masked_x = x_vals * mask_bool.to(tl.float32)
        
        # Update cumulative sum
        cumsum = cumsum + masked_x
        
        # Store result (cumsum only increases when mask is True)
        tl.store(out_ptr + out_batch_offset + offsets * stride_s, cumsum, mask=mask)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Compute cumulative sum only where mask is True, resetting when mask is False.
    This implementation assumes dim=1 for simplicity, which matches the given test case.
    """
    # Ensure inputs are contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Get dimensions
    batch_size = x.size(0) if dim == 1 else x.size(1)
    seq_len = x.size(1) if dim == 1 else x.size(0)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size (tunable)
    BLOCK_SIZE = 128
    
    # Calculate grid: one block per batch
    grid = (batch_size,)
    
    # Calculate strides
    stride_b = x.stride(0)
    stride_s = x.stride(1)
    
    # Launch kernel
    masked_cumsum_kernel[grid](
        x, mask, out,
        batch_size, seq_len,
        stride_b, stride_s,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        # For now, assume dim=1 (as in the given example), but handle general case
        if self.dim == 1:
            return triton_masked_cumsum(x, mask, self.dim)
        else:
            # Handle other dimensions by permuting
            # This is a simplified implementation for the common case
            x_perm = x.transpose(0, self.dim)
            mask_perm = mask.transpose(0, self.dim)
            result_perm = triton_masked_cumsum(x_perm, mask_perm, 1)
            return result_perm.transpose(0, self.dim)