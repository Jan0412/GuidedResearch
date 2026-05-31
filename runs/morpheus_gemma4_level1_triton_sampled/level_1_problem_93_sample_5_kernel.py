import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr,      # Pointer to input tensor x
    mask_ptr,   # Pointer to boolean mask tensor
    out_ptr,    # Pointer to output tensor
    n_cols,     # Number of elements along the cumsum dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element)
    row_idx = tl.program_id(0)
    
    # Calculate row offsets
    row_offset = row_idx * n_cols
    
    # Initialize the cumulative sum carry-over for the row
    prev_sum = 0.0
    
    # Process the row in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask_range = offsets < n_cols
        
        # Load values and masks for the current block
        x_vals = tl.load(x_ptr + row_offset + offsets, mask=mask_range, other=0.0)
        mask_vals = tl.load(mask_ptr + row_offset + offsets, mask=mask_range, other=False)
        
        # Apply mask: elements where mask is False become 0.0
        # Boolean mask_vals are automatically cast to float (0.0 or 1.0) during multiplication
        masked_vals = x_vals * mask_vals
        
        # Compute the local cumulative sum for the current block
        local_cumsum = tl.cumsum(masked_vals, axis=0)
        
        # Add the carry-over from previous blocks
        result = local_cumsum + prev_sum
        
        # Store the result back to global memory
        tl.store(out_ptr + row_offset + offsets, result, mask=mask_range)
        
        # Update the carry-over for the next block
        prev_sum += tl.sum(masked_vals, axis=0)

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor):
    """
    Triton wrapper for the masked cumulative sum operation.
    """
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for pointer arithmetic
    x = x.contiguous()
    mask = mask.contiguous()
    
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    # Block size for processing the cumsum dimension
    BLOCK_SIZE = 1024
    
    # Launch one program per row
    grid = (n_rows,)
    
    masked_cumsum_kernel[grid](
        x, mask, out, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a masked cumulative sum using a custom Triton kernel.
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
        # The Triton kernel is optimized for dim=1 (the last dimension in the provided case)
        # If dim is not the last dimension, the tensor would need to be transposed.
        if self.dim == 1 or (self.dim == -1 and x.ndim == 2):
            return triton_masked_cumsum(x, mask)
        else:
            # Fallback to PyTorch for other dimensions to maintain correctness
            return torch.cumsum(x * mask, dim=self.dim)