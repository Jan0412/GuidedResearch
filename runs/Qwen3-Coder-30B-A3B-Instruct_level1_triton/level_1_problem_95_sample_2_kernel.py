import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def log_softmax_kernel(
    inp_ptr,
    out_ptr,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate starting position for this row
    row_start = row_idx * n_cols
    
    # Initialize max and sum for numerical stability
    max_val = -float('inf')
    sum_val = 0.0
    
    # First pass: find max value
    for col in range(0, n_cols, BLOCK_SIZE):
        col_idx = col + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # Load input values
        inp_vals = tl.load(inp_ptr + row_start + col_idx, mask=mask, other=-float('inf'))
        
        # Update max
        max_val = tl.maximum(max_val, tl.max(inp_vals))
    
    # Second pass: compute log-sum-exp
    for col in range(0, n_cols, BLOCK_SIZE):
        col_idx = col + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # Load input values
        inp_vals = tl.load(inp_ptr + row_start + col_idx, mask=mask, other=0.0)
        
        # Compute exponentiated values with shift for numerical stability
        shifted_vals = inp_vals - max_val
        exp_vals = tl.exp(shifted_vals)
        
        # Accumulate sum
        sum_val += tl.sum(exp_vals)
    
    # Third pass: compute final log_softmax values
    for col in range(0, n_cols, BLOCK_SIZE):
        col_idx = col + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # Load input values
        inp_vals = tl.load(inp_ptr + row_start + col_idx, mask=mask, other=0.0)
        
        # Compute log_softmax with numerical stability
        shifted_vals = inp_vals - max_val
        log_sum_exp = tl.log(sum_val)
        out_vals = shifted_vals - log_sum_exp
        
        # Store results
        tl.store(out_ptr + row_start + col_idx, out_vals, mask=mask)

@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate starting positions
    logits_start = row_idx * n_cols
    target_idx = tl.load(targets_ptr + row_idx)
    
    # Load logits for this row
    logits = tl.load(logits_ptr + logits_start, mask=tl.arange(0, n_cols) < n_cols, other=-float('inf'))
    
    # Find max for numerical stability
    max_val = tl.max(logits)
    
    # Compute log-sum-exp
    shifted_logits = logits - max_val
    log_sum_exp = tl.log(tl.sum(tl.exp(shifted_logits)))
    
    # Compute negative log likelihood
    correct_logit = logits[target_idx]
    loss = log_sum_exp - correct_logit
    
    # Store result
    tl.store(out_ptr + row_idx, loss)

def triton_log_softmax(x: torch.Tensor):
    """Custom Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    GROUP_SIZE_M = 8
    
    # Grid calculation
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    log_softmax_kernel[grid](
        x,
        out,
        n_cols,
        n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return out

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Custom Triton implementation of cross entropy loss"""
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    out = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    cross_entropy_kernel[grid](
        logits,
        targets,
        out,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.mean()

class ModelNew(nn.Module):
    """
    An optimized model using custom Triton kernels for cross entropy loss computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use custom Triton kernel instead of torch.nn.functional.cross_entropy
        return triton_cross_entropy(predictions, targets)