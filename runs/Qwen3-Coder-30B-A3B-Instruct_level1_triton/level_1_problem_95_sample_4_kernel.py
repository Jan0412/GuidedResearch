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
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Calculate row start offset
    row_start = row_idx * n_cols
    
    # Load input row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * n_cols
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find max value in row for numerical stability
    max_val = tl.max(input_vals)
    
    # Compute exp(x - max_val) and sum
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals)
    
    # Compute log(sum_exp) and final log_softmax
    log_sum_exp = tl.log(sum_exp)
    output_vals = input_vals - max_val - log_sum_exp
    
    # Store results
    tl.store(output_ptr + offsets, output_vals, mask=mask)

@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Calculate row start offset
    row_start = row_idx * n_cols
    
    # Load logits and target
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * n_cols
    
    logits = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))
    target = tl.load(targets_ptr + row_idx, other=0)
    
    # Find max for numerical stability
    max_val = tl.max(logits)
    
    # Compute log_softmax
    exp_logits = tl.exp(logits - max_val)
    sum_exp = tl.sum(exp_logits)
    log_sum_exp = tl.log(sum_exp)
    log_probs = logits - max_val - log_sum_exp
    
    # Get negative log likelihood loss
    loss = -log_probs[target]
    
    # Store result
    tl.store(output_ptr + row_idx, loss, mask=row_idx < n_rows)

def triton_log_softmax(x: torch.Tensor):
    """Compute log_softmax using Triton kernel"""
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Prepare output
    output = torch.empty_like(x)
    
    # Grid configuration
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    log_softmax_kernel[grid](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Compute cross entropy using Triton kernel"""
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA"
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    
    # Prepare output
    output = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    # Grid configuration
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    cross_entropy_kernel[grid](logits, targets, output, n_rows, n_cols, BLOCK_SIZE=n_cols)
    
    # Return mean loss
    return output.mean()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use custom Triton kernel instead of torch.nn.functional.cross_entropy
        return triton_cross_entropy(predictions, targets)