import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr, 
    mask_ptr, 
    out_ptr, 
    S, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform masked cumulative sum along a single dimension.
    Each program handles one scan line (one row after reshaping).
    """
    # The program ID represents the index of the scan line (the 'batch' dimension)
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row for each tensor
    x_row_ptr = x_ptr + row_idx * S
    mask_row_ptr = mask_ptr + row_idx * S
    out_row_ptr = out_ptr + row_idx * S
    
    # Accumulator for the prefix sum across blocks
    acc = 0.0
    
    # Process the scan line in chunks of BLOCK_SIZE
    for i in range(0, S, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        # Mask to prevent out-of-bounds access at the end of the row
        mask = col_offsets < (S - i)
        
        # Load input values and the boolean mask
        x = tl.load(x_row_ptr + i + col_offsets, mask=mask, other=0.0)
        m = tl.load(mask_row_ptr + i + col_offsets, mask=mask, other=False)
        
        # Apply the mask: only keep values where mask is True
        val = tl.where(m, x, 0.0)
        
        # Compute local cumulative sum for the current block
        local_cumsum = tl.cumsum(val, axis=0)
        
        # Add the accumulator from previous blocks to the local cumsum
        global_cumsum = local_cumsum + acc
        
        # Store the result back to the output tensor
        tl.store(out_row_ptr + i + col_offsets, global_cumsum, mask=mask)
        
        # Update the accumulator with the total sum of the current block
        acc += tl.sum(val, axis=0)

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    Wrapper for the Triton masked cumsum kernel.
    Handles tensor reshaping and permutation to ensure the target dimension is the last one.
    """
    # Ensure inputs are on CUDA and contiguous
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    
    original_shape = x.shape
    S = original_shape[dim]
    
    # 1. Move the target dimension to the last position
    # This allows us to treat the tensor as a 2D (Others, S) matrix
    x_transposed = x.transpose(dim, -1).contiguous()
    mask_transposed = mask.transpose(dim, -1).contiguous()
    
    # 2. Reshape to 2D: (Batch, S)
    x_flat = x_transposed.view(-1, S)
    mask_flat = mask_transposed.view(-1, S)
    out_flat = torch.empty_like(x_flat)
    
    B_flat = x_flat.shape[0]
    BLOCK_SIZE = 1024 # Optimized block size for FP32 prefix sum
    
    # Define the grid: one program per scan line
    grid = (B_flat,)
    
    # Launch the kernel
    masked_cumsum_kernel[grid](
        x_flat, 
        mask_flat, 
        out_flat, 
        S, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 3. Reshape and transpose back to the original dimensions
    out = out_flat.view(x_transposed.shape).transpose(-1, dim)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using custom Triton kernels.
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