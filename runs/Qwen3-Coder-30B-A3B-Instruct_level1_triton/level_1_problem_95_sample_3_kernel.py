import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def log_softmax_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get row index
    row_idx = tl.program_id(0)
    
    # Each row is processed by one thread block
    if row_idx >= n_rows:
        return
    
    # Calculate row start position
    row_start = row_idx * n_cols
    
    # Load input data for this row
    input_row = tl.load(input_ptr + row_start + tl.arange(0, BLOCK_SIZE), mask=(tl.arange(0, BLOCK_SIZE) < n_cols))
    
    # Find max value in the row
    max_val = tl.max(input_row)
    
    # Compute exp(x - max_val) and sum
    exp_sum = 0.0
    exp_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for i in range(0, n_cols, BLOCK_SIZE):
        # Load chunk
        chunk_indices = i + tl.arange(0, BLOCK_SIZE)
        mask = chunk_indices < n_cols
        chunk = tl.load(input_ptr + row_start + chunk_indices, mask=mask)
        
        # Compute exp(x - max_val)
        exp_chunk = tl.exp(chunk - max_val)
        tl.store(output_ptr + row_start + chunk_indices, exp_chunk, mask=mask)
        
        # Add to sum
        exp_sum += tl.sum(exp_chunk * mask.to(tl.float32))
    
    # Compute log(sum(exp)) 
    log_sum = tl.log(exp_sum)
    
    # Compute final log_softmax
    for i in range(0, n_cols, BLOCK_SIZE):
        chunk_indices = i + tl.arange(0, BLOCK_SIZE)
        mask = chunk_indices < n_cols
        exp_val = tl.load(output_ptr + row_start + chunk_indices, mask=mask)
        log_softmax_val = exp_val - log_sum - max_val
        tl.store(output_ptr + row_start + chunk_indices, log_softmax_val, mask=mask)

@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get row index
    row_idx = tl.program_id(0)
    
    # Each row is processed by one thread block
    if row_idx >= n_rows:
        return
    
    # Calculate row start positions
    row_start_logits = row_idx * n_cols
    target_idx = tl.load(targets_ptr + row_idx)
    
    # Load logits for this row
    logits_row = tl.load(logits_ptr + row_start_logits + tl.arange(0, BLOCK_SIZE), mask=(tl.arange(0, BLOCK_SIZE) < n_cols))
    
    # Find max value in the row
    max_val = tl.max(logits_row)
    
    # Compute log-sum-exp
    exp_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        chunk_indices = i + tl.arange(0, BLOCK_SIZE)
        mask = chunk_indices < n_cols
        chunk = tl.load(logits_ptr + row_start_logits + chunk_indices, mask=mask)
        exp_chunk = tl.exp(chunk - max_val)
        exp_sum += tl.sum(exp_chunk * mask.to(tl.float32))
    
    log_sum_exp = tl.log(exp_sum)
    
    # Compute negative log likelihood
    target_log_prob = tl.load(logits_ptr + row_start_logits + target_idx) - log_sum_exp - max_val
    loss = -target_log_prob
    
    # Store result
    tl.store(output_ptr + row_idx, loss)

def triton_log_softmax(input_tensor):
    """Custom Triton implementation of log_softmax"""
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    
    n_rows, n_cols = input_tensor.shape
    output = torch.empty_like(input_tensor)
    
    # Use block size that works well for most cases
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    log_softmax_kernel[grid](
        input_tensor,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def triton_cross_entropy(logits, targets):
    """Custom Triton implementation of cross entropy loss"""
    assert logits.is_cuda, "Logits tensor must be on CUDA"
    assert targets.is_cuda, "Targets tensor must be on CUDA"
    
    n_rows, n_cols = logits.shape
    output = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    # Use block size that works well for most cases
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    cross_entropy_kernel[grid](
        logits,
        targets,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean loss
    return output.mean()

class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks
    using custom Triton kernels for optimization.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Using custom Triton kernel instead of PyTorch's built-in cross_entropy
        return triton_cross_entropy(predictions, targets)

# For compatibility with original interface
batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []