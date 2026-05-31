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
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    # Load input row
    row_start = row_id * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Process in chunks
    for chunk_start in range(0, n_cols, BLOCK_SIZE):
        chunk_offsets = chunk_start + offsets
        chunk_mask = chunk_offsets < n_cols
        
        # Load input values
        inp_vals = tl.load(inp_ptr + chunk_offsets, mask=chunk_mask, other=-float('inf'))
        
        # Compute max for numerical stability
        max_val = tl.max(inp_vals)
        
        # Compute exp(x - max) and sum
        exp_vals = tl.exp(inp_vals - max_val)
        sum_exp = tl.sum(exp_vals)
        
        # Compute log(sum_exp) + max_val
        log_sum_exp = tl.log(sum_exp) + max_val
        
        # Compute log_softmax
        log_softmax_vals = inp_vals - log_sum_exp
        
        # Store results
        tl.store(out_ptr + chunk_offsets, log_softmax_vals, mask=chunk_mask)

@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    # Load target for this row
    target = tl.load(targets_ptr + row_id)
    
    # Load logits for this row
    row_start = row_id * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load logits
    logits_vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(logits_vals)
    
    # Compute exp(x - max) and sum
    exp_vals = tl.exp(logits_vals - max_val)
    sum_exp = tl.sum(exp_vals)
    
    # Compute log(sum_exp) + max_val
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Get log probability of target class
    target_log_prob = logits_vals[target] - log_sum_exp
    
    # Store negative log likelihood loss
    loss = -target_log_prob
    tl.store(loss_ptr + row_id, loss)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Grid configuration
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of cross entropy loss"""
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    # Grid configuration
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    cross_entropy_kernel[grid](logits, targets, loss, n_rows, n_cols, BLOCK_SIZE=1024)
    
    return loss.mean()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use custom Triton kernels instead of torch.nn.functional.cross_entropy
        return triton_cross_entropy(predictions, targets)