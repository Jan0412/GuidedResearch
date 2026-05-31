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
    # Calculate the starting column for this thread block
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # Shared memory for reduction operations
    shared_max = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    shared_sum = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Initialize max and sum for this row
    row_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    row_sum = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
    
    # Process elements in chunks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual column indices
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # Load input values
        input_vals = tl.load(input_ptr + row_idx * n_cols + col_idx, mask=mask, other=0.0)
        
        # Compute max for this chunk
        chunk_max = tl.max(input_vals, axis=0)
        row_max = tl.maximum(row_max, chunk_max)
        
        # Compute exp and sum for this chunk
        exp_vals = tl.exp(input_vals - row_max)
        row_sum += tl.sum(exp_vals, axis=0)
        
        # Store intermediate results in shared memory
        tl.store(shared_max + tl.arange(0, BLOCK_SIZE), row_max, mask=mask)
        tl.store(shared_sum + tl.arange(0, BLOCK_SIZE), row_sum, mask=mask)
    
    # Reduce across all blocks for this row
    # For simplicity, we assume single block per row for this implementation
    # In practice, this would involve additional reduction steps
    
    # Compute final softmax values
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # Load input values again
        input_vals = tl.load(input_ptr + row_idx * n_cols + col_idx, mask=mask, other=0.0)
        
        # Compute final softmax
        softmax_vals = tl.exp(input_vals - row_max) / row_sum
        
        # Store output
        tl.store(output_ptr + row_idx * n_cols + col_idx, softmax_vals, mask=mask)

# More efficient version using proper reduction
@triton.jit
def softmax_kernel_optimized(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Shared memory for reduction
    shared_max = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    shared_sum = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Load row data
    row_start = row_idx * n_cols
    row_input = tl.load(input_ptr + row_start + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
    
    # Find maximum value in the row
    row_max = tl.max(row_input, axis=0)
    
    # Compute exponentials and sum
    exp_vals = tl.exp(row_input - row_max)
    row_sum = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    softmax_vals = exp_vals / row_sum
    
    # Store result
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), softmax_vals, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

# Even more optimized version with better memory access patterns
@triton.jit
def softmax_kernel_final(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Shared memory for reduction
    row_max = tl.shared_memory(dtype=tl.float32, size=1)
    row_sum = tl.shared_memory(dtype=tl.float32, size=1)
    
    # Process the row
    row_start = row_idx * n_cols
    
    # Load entire row (assuming it fits within shared memory or we handle it properly)
    # For very large rows, we'd need a loop but let's keep it simple for now
    
    # Load first chunk
    chunk_input = tl.load(input_ptr + row_start + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
    
    # Compute max and sum
    chunk_max = tl.max(chunk_input, axis=0)
    chunk_sum = tl.sum(tl.exp(chunk_input - chunk_max), axis=0)
    
    # Store reductions
    if tl.program_id(1) == 0:
        row_max[0] = chunk_max
        row_sum[0] = chunk_sum
    
    # Synchronize to make sure reductions are available
    tl.sync()
    
    # Compute final softmax
    chunk_output = tl.exp(chunk_input - row_max[0]) / row_sum[0]
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), chunk_output, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

# Corrected and simplified approach for better performance
@triton.jit
def softmax_kernel_simple(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get row index
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Shared memory for reduction
    shared_max = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    shared_sum = tl.shared_memory(dtype=tl.float32, size=BLOCK_SIZE)
    
    # Load data in chunks
    max_val = float('-inf')
    sum_val = 0.0
    
    # Process entire row in chunks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate current offset
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_cols
        
        # Load input values
        vals = tl.load(input_ptr + row_start + offset, mask=mask, other=float('-inf'))
        
        # Compute local max
        local_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, local_max)
    
    # Second pass to compute sum
    sum_val = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_cols
        
        vals = tl.load(input_ptr + row_start + offset, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_val += tl.sum(exp_vals, axis=0)
    
    # Final pass to compute softmax
    for i in range(0, n_cols, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_cols
        
        vals = tl.load(input_ptr + row_start + offset, mask=mask, other=0.0)
        softmax_vals = tl.exp(vals - max_val) / sum_val
        tl.store(output_ptr + row_start + offset, softmax_vals, mask=mask)

# Final optimized version using proper reduction approach
@triton.jit
def softmax_kernel_best(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get row index
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Load full row
    row_input = tl.load(input_ptr + row_start + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=0.0)
    
    # Compute max value
    row_max = tl.max(row_input, axis=0)
    
    # Compute exponential and sum
    exp_vals = tl.exp(row_input - row_max)
    row_sum = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    softmax_vals = exp_vals / row_sum
    
    # Store result
    tl.store(output_ptr + row_start + tl.arange(0, BLOCK_SIZE), softmax_vals, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

def triton_softmax(x: torch.Tensor):
    """
    Triton-based softmax implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Parameters
    batch_size, n_cols = x.shape
    BLOCK_SIZE = 1024  # Adjusted block size
    
    # Grid configuration
    grid = (batch_size, 1)
    
    # Launch kernel
    softmax_kernel_best[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model with Triton-based softmax.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-optimized Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)