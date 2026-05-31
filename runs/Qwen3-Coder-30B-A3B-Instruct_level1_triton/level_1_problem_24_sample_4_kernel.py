import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def logsoftmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Each program processes one row
    if row_idx >= n_rows:
        return
    
    # Calculate the starting offset for this row
    input_row_ptr = input_ptr + row_idx * n_cols
    output_row_ptr = output_ptr + row_idx * n_cols
    
    # Load the row into shared memory
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Process in chunks to handle large rows
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets and mask for this chunk
        chunk_offsets = i + offsets
        chunk_mask = (chunk_offsets < n_cols) & mask
        
        # Load input values
        input_vals = tl.load(input_row_ptr + chunk_offsets, mask=chunk_mask, other=-float('inf'))
        
        # Find max value in this chunk for numerical stability
        max_val = tl.max(input_vals)
        
        # Compute exp(x - max_val) and sum
        exp_vals = tl.exp(input_vals - max_val)
        sum_exp = tl.sum(exp_vals)
        
        # Compute log(sum_exp) and store final result
        log_sum_exp = tl.log(sum_exp) + max_val
        output_vals = input_vals - log_sum_exp
        
        # Store results back
        tl.store(output_row_ptr + chunk_offsets, output_vals, mask=chunk_mask)

def triton_logsoftmax(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of LogSoftmax with numerical stability.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    n_rows = x.shape[0]
    n_cols = x.shape[dim]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Choose appropriate block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (n_rows,)
    
    # Launch kernel
    logsoftmax_kernel[grid](
        x,
        out,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for LogSoftmax.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return triton_logsoftmax(x, self.dim)