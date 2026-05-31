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
    
    # Calculate starting offset for this row
    row_start = row_id * n_cols
    
    # Load data for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Process in blocks to handle large rows
    max_val = -float('inf')
    sum_exp = 0.0
    
    # First pass: find max value
    for i in range(0, n_cols, BLOCK_SIZE):
        block_offsets = i + offsets
        block_mask = mask & (block_offsets < n_cols)
        vals = tl.load(inp_ptr + row_start + block_offsets, mask=block_mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(vals))
    
    # Second pass: compute sum of exponentials
    for i in range(0, n_cols, BLOCK_SIZE):
        block_offsets = i + offsets
        block_mask = mask & (block_offsets < n_cols)
        vals = tl.load(inp_ptr + row_start + block_offsets, mask=block_mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals)
    
    # Third pass: compute final log_softmax values
    for i in range(0, n_cols, BLOCK_SIZE):
        block_offsets = i + offsets
        block_mask = mask & (block_offsets < n_cols)
        vals = tl.load(inp_ptr + row_start + block_offsets, mask=block_mask, other=0.0)
        out_vals = vals - max_val - tl.log(sum_exp)
        tl.store(out_ptr + row_start + block_offsets, out_vals, mask=block_mask)

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
    
    # Calculate starting offset for this row
    row_start = row_id * n_cols
    target_idx = tl.load(targets_ptr + row_id)
    
    # Find max for numerical stability
    max_val = -float('inf')
    for i in range(0, n_cols, BLOCK_SIZE):
        block_offsets = i + tl.arange(0, BLOCK_SIZE)
        block_mask = (block_offsets < n_cols) & (block_offsets >= 0)
        vals = tl.load(logits_ptr + row_start + block_offsets, mask=block_mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(vals))
    
    # Compute log-sum-exp
    sum_exp = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        block_offsets = i + tl.arange(0, BLOCK_SIZE)
        block_mask = (block_offsets < n_cols) & (block_offsets >= 0)
        vals = tl.load(logits_ptr + row_start + block_offsets, mask=block_mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals)
    
    # Compute negative log likelihood
    log_sum_exp = max_val + tl.log(sum_exp)
    target_log_prob = tl.load(logits_ptr + row_start + target_idx) - log_sum_exp
    loss_val = -target_log_prob
    
    tl.store(loss_ptr + row_id, loss_val)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_rows,)
    
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of cross entropy loss"""
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_rows,)
    
    cross_entropy_kernel[grid](logits, targets, loss, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return loss.mean()

class ModelNew(nn.Module):
    """
    An optimized model using Triton kernels for cross entropy loss computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our Triton-based cross entropy implementation
        return triton_cross_entropy(predictions, targets)