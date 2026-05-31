import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting column index for this thread block
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # Shared memory for reduction
    shared_max = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    shared_sum = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Initialize max and sum for this row
    row_max = tl.full([1], float('-inf'), dtype=tl.float32)
    row_sum = tl.full([1], 0.0, dtype=tl.float32)
    
    # Process columns in chunks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual column indices
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid columns
        mask = col_idx < n_cols
        
        # Load input values
        input_vals = tl.load(input_ptr + row_idx * n_cols + col_idx, mask=mask, other=float('-inf'))
        
        # Compute max of current chunk
        chunk_max = tl.max(input_vals, axis=0)
        row_max = tl.maximum(row_max, chunk_max)
    
    # Broadcast row_max to all threads in the block
    row_max = tl.broadcast_to(row_max, [BLOCK_SIZE])
    
    # Reset row_sum
    row_sum = tl.full([1], 0.0, dtype=tl.float32)
    
    # Compute exponentials and sum
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        input_vals = tl.load(input_ptr + row_idx * n_cols + col_idx, mask=mask, other=0.0)
        
        # Compute exp(x - max)
        exp_vals = tl.exp(input_vals - row_max)
        
        # Store results in shared memory for reduction
        tl.store(shared_max + tl.arange(0, BLOCK_SIZE), exp_vals, mask=mask)
        
        # Reduce within block
        local_sum = tl.sum(exp_vals, axis=0)
        row_sum += local_sum
    
    # Broadcast row_sum to all threads
    row_sum = tl.broadcast_to(row_sum, [BLOCK_SIZE])
    
    # Final pass to compute final softmax
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        input_vals = tl.load(input_ptr + row_idx * n_cols + col_idx, mask=mask, other=0.0)
        
        # Compute final softmax values
        exp_vals = tl.exp(input_vals - row_max)
        softmax_vals = exp_vals / row_sum
        
        # Store results
        tl.store(output_ptr + row_idx * n_cols + col_idx, softmax_vals, mask=mask)

def triton_softmax(x: torch.Tensor):
    """
    Triton implementation of softmax with optimized memory access patterns
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, n_cols = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid dimensions
    grid = (
        batch_size,
        (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    
    # Launch kernel
    softmax_kernel[grid](
        x,
        output,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for softmax operation
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softmax activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)