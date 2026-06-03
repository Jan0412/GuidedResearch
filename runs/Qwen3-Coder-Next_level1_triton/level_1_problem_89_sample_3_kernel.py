import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Calculate base pointers for this batch
    x_offset = batch_id * seq_len
    out_offset = batch_id * seq_len
    
    # Process the sequence in blocks
    for start in range(0, seq_len, BLOCK_SIZE):
        # Compute end index for this block
        end = tl.minimum(start + BLOCK_SIZE, seq_len)
        length = end - start
        
        # Create offsets for current block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len
        
        # Load input values
        x = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0)
        
        # Compute cumulative sum in this block
        # We need to accumulate across blocks, so we'll use a scan approach
        # First compute prefix sum within this block
        cumsum_block = tl.cumsum(x, axis=0)
        
        # For blocks after the first, add the sum of all previous blocks
        if start > 0:
            # Load the cumulative sum up to the start of this block
            prev_sum = tl.load(out_ptr + out_offset + start - 1, mask=(start - 1) < seq_len)
            cumsum_block = cumsum_block + prev_sum
        
        # Store the result
        tl.store(out_ptr + out_offset + offsets, cumsum_block, mask=mask)


def triton_cumsum(x: torch.Tensor, dim: int):
    """
    Triton-based cumulative sum implementation optimized for FP32.
    
    Args:
        x: Input tensor (FP32)
        dim: Dimension along which to compute cumulative sum
        
    Returns:
        Tensor with cumulative sum along specified dimension
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Ensure we're working with FP32 for precision
    if x.dtype != torch.float32:
        x = x.float()
    
    # Get dimensions
    batch_size = x.size(0) if dim == 1 else x.size(1) if dim == 0 else x.numel() // (x.size(0) if 0 in [dim, 1] else x.size(1))
    seq_len = x.size(dim)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # For 2D tensors, handle dim=0 or dim=1
    if x.dim() == 2:
        if dim == 1:
            # Batch dimension is first, sequence is second
            batch_size = x.size(0)
            seq_len = x.size(1)
        else:  # dim == 0
            # Need to transpose for efficiency, then transpose back
            x = x.t()
            batch_size = x.size(0)
            seq_len = x.size(1)
    
    # Set block size - tuned for memory bandwidth and occupancy
    BLOCK_SIZE = 128
    
    # Create grid: one program per batch
    grid = (batch_size,)
    
    # Launch kernel
    cumsum_kernel[grid](x, out, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)
    
    # If we transposed earlier (dim=0 case), transpose back
    if dim == 0 and x.dim() == 2:
        out = out.t()
    
    return out


class ModelNew(nn.Module):
    """
    Optimized Scan model using Triton kernel for cumulative sum operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass for the optimized Scan model using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        return triton_cumsum(x, self.dim)