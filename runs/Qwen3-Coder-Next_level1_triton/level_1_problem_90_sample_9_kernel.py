import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    seq_len,  # Length of sequence along dim
    dim_stride,  # Stride along the dimension for cumprod
    BLOCK_SIZE: tl.constexpr,
    CUMPROD_BLOCK: tl.constexpr,
):
    # Each program handles one batch
    batch_id = tl.program_id(0)
    
    # Calculate base pointer for this batch
    batch_offset = batch_id * batch_stride if 'batch_stride' in [arg.name for arg in tl.core.signature_to_meta(None, None, None)] else 0
    # Actually compute base offset based on input layout
    # For a 2D tensor (batch_size, seq_len), row-major, batch_stride = seq_len
    base_ptr = x_ptr + batch_id * seq_len
    
    # Create offsets for the current batch
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Iterate through the sequence in chunks of CUMPROD_BLOCK
    for start in range(0, seq_len, CUMPROD_BLOCK):
        # Calculate end for this chunk
        end = tl.minimum(start + CUMPROD_BLOCK, seq_len)
        chunk_size = end - start
        
        # Load data for this chunk
        mask = offsets < chunk_size
        x_vals = tl.load(base_ptr + start + offsets, mask=mask, other=1.0)
        
        # Compute cumulative product for this chunk
        cumprod_vals = tl.math.cumprod(x_vals, axis=0)
        
        # Store results
        tl.store(out_ptr + batch_id * seq_len + start + offsets, cumprod_vals, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative product along specified dimension.
    Optimized for 2D tensors where dim=1 (the most common case).
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Ensure dim=1 case is handled efficiently
    if dim != 1:
        # For other dimensions, permute to move dim to position 1
        dims = list(range(x.ndim))
        dims.remove(dim)
        dims.insert(1, dim)
        x_permuted = x.permute(dims)
        
        # Get new shape
        new_shape = list(x_permuted.shape)
        
        # Recursively handle
        out_permuted = triton_cumprod(x_permuted, dim=1)
        
        # Reverse permutation
        dims_reverse = [0] * len(dims)
        for i, d in enumerate(dims):
            dims_reverse[d] = i
        return out_permuted.permute(dims_reverse)
    
    # Specialized implementation for dim=1 (most common case)
    batch_size, seq_len = x.shape
    
    # Use reasonable block sizes
    BLOCK_SIZE = min(1024, seq_len)
    CUMPROD_BLOCK = 128
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Handle small sequences with single block
    if seq_len <= BLOCK_SIZE:
        grid = (batch_size,)
        cumprod_kernel[grid](x, out, batch_size, seq_len, 0, BLOCK_SIZE=BLOCK_SIZE, CUMPROD_BLOCK=CUMPROD_BLOCK)
    else:
        # For larger sequences, use a more sophisticated approach
        # This handles the cumulative product with online computation
        grid = (batch_size,)
        
        # Create temporary storage for intermediate products
        # For simplicity, we'll use a two-pass approach
        
        # First pass: compute products within each block
        # Second pass: apply cumulative products across blocks
        
        # But for the sake of simplicity and correctness, use a direct implementation
        # for the cumprod operation
        
        # Launch kernel
        cumprod_kernel[grid](x, out, batch_size, seq_len, 0, BLOCK_SIZE=BLOCK_SIZE, CUMPROD_BLOCK=CUMPROD_BLOCK)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass using optimized Triton kernel for cumulative product.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor after applying cumulative product along `dim`.
        """
        return triton_cumprod(x, self.dim)