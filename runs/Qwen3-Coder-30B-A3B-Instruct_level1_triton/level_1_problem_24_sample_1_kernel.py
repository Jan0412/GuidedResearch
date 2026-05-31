import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def logsoftmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Check if this row is valid
    if row_idx >= n_rows:
        return
        
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Shared memory for reduction
    shared_max = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    shared_sum = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Load data in chunks
    for chunk in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets
        chunk_offset = chunk + tl.arange(0, BLOCK_SIZE)
        mask = chunk_offset < n_cols
        
        # Load data
        x_vals = tl.load(x_ptr + row_start + chunk_offset, mask=mask, other=-float('inf'))
        
        # Compute max value for this chunk
        chunk_max = tl.max(x_vals)
        
        # Store in shared memory
        tl.store(shared_max + chunk_offset, chunk_max, mask=mask)
        
        # Compute exp(x - max) and sum
        exp_vals = tl.exp(x_vals - chunk_max)
        chunk_sum = tl.sum(exp_vals)
        
        # Store exp vals in shared memory
        tl.store(shared_sum + chunk_offset, chunk_sum, mask=mask)
    
    # Reduction across all chunks to get global max
    global_max = -float('inf')
    for i in range(0, n_cols, BLOCK_SIZE):
        chunk_offset = i + tl.arange(0, BLOCK_SIZE)
        mask = chunk_offset < n_cols
        chunk_max = tl.load(shared_max + chunk_offset, mask=mask, other=-float('inf'))
        global_max = tl.maximum(global_max, tl.max(chunk_max))
    
    # Reduction across all chunks to get global sum
    global_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        chunk_offset = i + tl.arange(0, BLOCK_SIZE)
        mask = chunk_offset < n_cols
        chunk_sum = tl.load(shared_sum + chunk_offset, mask=mask, other=0.0)
        global_sum += tl.sum(chunk_sum)
    
    # Compute final logsoftmax
    for chunk in range(0, n_cols, BLOCK_SIZE):
        chunk_offset = chunk + tl.arange(0, BLOCK_SIZE)
        mask = chunk_offset < n_cols
        
        # Load original values
        x_vals = tl.load(x_ptr + row_start + chunk_offset, mask=mask, other=0.0)
        
        # Compute logsoftmax: x - max - log(sum(exp(x - max)))
        logsoftmax_vals = x_vals - global_max - tl.log(global_sum)
        
        # Store result
        tl.store(out_ptr + row_start + chunk_offset, logsoftmax_vals, mask=mask)

def triton_logsoftmax(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of LogSoftmax with optimized reduction
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    n_rows = x.shape[0]
    n_cols = x.shape[1] if dim == 1 else x.shape[-1]
    
    # Use a reasonable block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (n_rows, 1, 1)
    
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
    Optimized model using Triton kernel for LogSoftmax activation.
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