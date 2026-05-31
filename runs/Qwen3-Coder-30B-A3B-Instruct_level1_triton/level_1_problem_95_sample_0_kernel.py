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
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    # Calculate the starting position for this row
    row_start = row_id * n_cols
    
    # Load input values for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_id + 1) * n_cols
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find max value for numerical stability
    max_val = tl.max(input_vals)
    
    # Compute exp(x - max_val) and sum
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals)
    
    # Compute log(sum_exp) and final result
    log_sum_exp = tl.log(sum_exp)
    result = input_vals - max_val - log_sum_exp
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

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
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    # Calculate the starting position for this row
    row_start = row_id * n_cols
    
    # Load logits and target for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_id + 1) * n_cols
    
    logits_vals = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))
    target = tl.load(targets_ptr + row_id, mask=True, other=0)
    
    # Compute log_softmax
    max_val = tl.max(logits_vals)
    exp_vals = tl.exp(logits_vals - max_val)
    sum_exp = tl.sum(exp_vals)
    log_sum_exp = tl.log(sum_exp)
    log_probs = logits_vals - max_val - log_sum_exp
    
    # Get negative log likelihood loss
    loss = -log_probs[target]
    
    # Store result
    tl.store(output_ptr + row_id, loss, mask=True)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine the number of blocks needed
    grid = lambda meta: (n_rows,)
    
    # Launch the Triton kernel
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of cross entropy loss"""
    assert logits.is_cuda, "Logits must be on CUDA."
    assert targets.is_cuda, "Targets must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    
    # Prepare output tensor
    out = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    # Determine the number of blocks needed
    grid = lambda meta: (n_rows,)
    
    # Launch the Triton kernel
    cross_entropy_kernel[grid](logits, targets, out, n_rows, n_cols, BLOCK_SIZE=n_cols)
    return out.mean()

class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks
    using custom Triton kernels for optimization.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use Triton implementation instead of PyTorch's built-in function
        return triton_cross_entropy(predictions, targets)