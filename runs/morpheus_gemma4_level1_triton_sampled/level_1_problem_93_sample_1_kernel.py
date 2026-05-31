import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr,      # Pointer to input tensor
    mask_ptr,   # Pointer to mask tensor
    out_ptr,    # Pointer to output tensor
    n_dim,      # Size of the dimension to sum over
    n_other,    # Total number of elements in other dimensions
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for masked cumulative sum.
    Each program handles one 'row' (one sequence along the dimension being summed).
    """
    # Program ID represents the index of the sequence we are processing
    row_idx = tl.program_id(0)
    
    # Create offsets for the dimension being summed
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_dim
    
    # Calculate pointers for the current row
    # The input is assumed to be reshaped to (n_other, n_dim)
    x_row_ptr = x_ptr + row_idx * n_dim
    m_row_ptr = mask_ptr + row_idx * n_dim
    o_row_ptr = out_ptr + row_idx * n_dim
    
    # Load the data and the mask
    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(m_row_ptr + offsets, mask=mask, other=0)
    
    # Apply the mask: elements where mask is False (0) are zeroed out
    # Casting mask to float32 to perform element-wise multiplication
    val = x * m.to(tl.float32)
    
    # Perform the cumulative sum along the loaded block
    # tl.cumsum is highly optimized for power-of-2 block sizes
    res = tl.cumsum(val, axis=0)
    
    # Store the result back to global memory
    tl.store(o_row_ptr + offsets, res, mask=mask)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Wrapper for the masked_cumsum_kernel.
    Handles tensor reshaping to ensure the summation dimension is the last one
    and contiguous, allowing the kernel to process rows in parallel.
    """
    # Ensure inputs are on CUDA
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    
    orig_shape = x.shape
    ndim = x.dim()
    # Normalize dimension (handle negative indices)
    dim = dim if dim >= 0 else ndim + dim
    
    # To make the kernel generic for any dimension, we move the target dim to the end
    # and flatten all other dimensions into one.
    # x_permuted shape: (batch_size * other_dims..., n_dim)
    x_permuted = x.movedim(dim, -1).contiguous()
    mask_permuted = mask.movedim(dim, -1).contiguous()
    
    n_dim = orig_shape[dim]
    n_other = x_permuted.numel() // n_dim
    
    # Prepare output tensor
    out = torch.empty_like(x_permuted)
    
    # BLOCK_SIZE must be a power of 2 for Triton's tl.cumsum
    BLOCK_SIZE = 1 << (n_dim - 1).bit_length()
    
    # Grid: one program per 'row' (sequence along the dim)
    grid = (n_other,)
    
    masked_cumsum_kernel[grid](
        x_permuted, 
        mask_permuted, 
        out, 
        n_dim, 
        n_other, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Restore original shape and dimension order
    out = out.view(x_permuted.shape).movedim(-1, dim)
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Model that performs a masked cumulative sum
    using a custom Triton kernel for fusion and speedup.
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
        # Use the Triton-optimized masked cumsum implementation
        return triton_masked_cumsum(x, mask, self.dim)