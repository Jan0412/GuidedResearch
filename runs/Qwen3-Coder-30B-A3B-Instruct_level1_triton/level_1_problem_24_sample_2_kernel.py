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
    
    # Check if we're within bounds
    if row_idx >= n_rows:
        return
    
    # Calculate the starting position for this row
    input_row_start = row_idx * n_cols
    output_row_start = row_idx * n_cols
    
    # Load a block of data for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Process the row in chunks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets for this chunk
        chunk_offsets = i + offsets
        chunk_mask = chunk_offsets < n_cols
        
        # Load input values for this chunk
        input_vals = tl.load(input_ptr + input_row_start + chunk_offsets, 
                           mask=chunk_mask, other=-float('inf'))
        
        # Find max value in the chunk for numerical stability
        max_val = tl.max(input_vals)
        
        # Subtract max and exponentiate
        exp_vals = tl.exp(input_vals - max_val)
        
        # Compute sum of exponentials
        sum_exp = tl.sum(exp_vals)
        
        # Compute log(sum_exp) and then log_softmax
        log_sum_exp = tl.log(sum_exp)
        log_softmax_vals = input_vals - max_val - log_sum_exp
        
        # Store results
        tl.store(output_ptr + output_row_start + chunk_offsets,
                log_softmax_vals, mask=chunk_mask)

def triton_logsoftmax(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of LogSoftmax with numerical stability
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Ensure we're working with float32
    if x.dtype != torch.float32:
        x = x.float()
    
    batch_size, dim_size = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Choose block size based on problem size
    BLOCK_SIZE = 1024
    
    # Determine grid size
    grid = (batch_size, 1, 1)
    
    # Launch kernel
    logsoftmax_kernel[grid](
        x,
        out,
        batch_size,
        dim_size,
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
        Applies LogSoftmax activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return triton_logsoftmax(x, self.dim)