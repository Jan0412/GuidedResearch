import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def masked_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    mask_ptr,  # Mask tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    stride,  # Stride between elements along the accumulation dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one "row" (all elements along the accumulation dimension)
    row_idx = tl.program_id(0)
    
    # Calculate base offsets for this row
    base_offset = row_idx * stride
    
    # Initialize running sum
    running_sum = tl.zeros((1,), dtype=tl.float32)
    
    # Process in blocks
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_idx in range(num_blocks):
        # Calculate global offset for this block
        offsets = base_offset + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_block = offsets < (row_idx + 1) * stride
        
        # Load x and mask values
        x_val = tl.load(x_ptr + offsets, mask=mask_block, other=0.0)
        mask_val = tl.load(mask_ptr + offsets, mask=mask_block, other=0).to(tl.float32)
        
        # Compute element-wise product
        product = x_val * mask_val
        
        # Update running sum for masked elements
        condition = mask_val > 0.5  # Convert to boolean condition
        running_sum = tl.where(condition, running_sum + product, running_sum)
        
        # Store result
        tl.store(out_ptr + offsets, running_sum, mask=mask_block)


@triton.jit
def masked_cumsum_fused_kernel(
    x_ptr,  # Input tensor pointer
    mask_ptr,  # Mask tensor pointer  
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of rows
    seq_len,  # Length of sequence along accumulation dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Base offset for this row
    base_offset = row_idx * seq_len
    
    # Initialize running sum
    running_sum = 0.0
    
    # Process elements sequentially within the row
    for i in range(seq_len):
        offset = base_offset + i
        x_val = tl.load(x_ptr + offset)
        mask_val = tl.load(mask_ptr + offset).to(tl.float32)
        
        # Only accumulate if mask is True
        if mask_val > 0.5:
            running_sum += x_val
            
        # Store the current running sum
        tl.store(out_ptr + offset, running_sum)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Compute masked cumulative sum along specified dimension.
    
    Args:
        x: Input tensor
        mask: Boolean mask tensor
        dim: Dimension along which to compute cumulative sum
        
    Returns:
        Tensor with masked cumulative sum
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    assert x.shape == mask.shape, "Input and mask must have same shape."
    
    # Ensure contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    shape = x.shape
    n_dims = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim += n_dims
    
    # Calculate stride and sizes
    stride = 1
    for i in range(dim + 1, n_dims):
        stride *= shape[i]
    
    seq_len = shape[dim]
    batch_size = x.numel() // seq_len
    
    # For 1D case (simplest), use direct approach
    if n_dims == 1:
        n_rows = 1
        BLOCK_SIZE = 256
        grid = (n_rows,)
        masked_cumsum_kernel[grid](x, mask, out, seq_len, seq_len, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # Use fused kernel for better performance with sequential dependency
        BLOCK_SIZE = 128
        grid = (batch_size,)
        
        # Calculate sequence length and batch size for kernel
        seq_dim_size = shape[dim]
        other_size = batch_size
        
        # For better performance, use the sequential kernel which handles
        # the inherent data dependency of cumulative sum
        masked_cumsum_fused_kernel[grid](x, mask, out, batch_size, seq_dim_size, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using custom Triton kernels.
    
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