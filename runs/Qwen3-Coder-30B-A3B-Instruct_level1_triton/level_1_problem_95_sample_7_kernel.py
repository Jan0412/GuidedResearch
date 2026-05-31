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
    
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Load the row data
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Initialize max_val to negative infinity
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    # First pass: find maximum value in the row
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + offsets
        col_mask = col_offsets < n_cols
        input_vals = tl.load(input_ptr + row_start + col_offsets, mask=col_mask, other=float('-inf'))
        max_val = tl.maximum(max_val, input_vals)
    
    # Reduce to find global max across all elements in row
    max_val = tl.max(max_val, axis=0)
    
    # Second pass: compute sum of exponentials
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + offsets
        col_mask = col_offsets < n_cols
        input_vals = tl.load(input_ptr + row_start + col_offsets, mask=col_mask, other=0.0)
        exp_vals = tl.exp(input_vals - max_val)
        sum_exp += exp_vals
    
    # Reduce sum_exp to scalar
    sum_exp = tl.sum(sum_exp, axis=0)
    
    # Third pass: compute log_softmax
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + offsets
        col_mask = col_offsets < n_cols
        input_vals = tl.load(input_ptr + row_start + col_offsets, mask=col_mask, other=0.0)
        output_vals = input_vals - max_val - tl.log(sum_exp)
        tl.store(output_ptr + row_start + col_offsets, output_vals, mask=col_mask)

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
    
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Load target index
    target_idx = tl.load(targets_ptr + row_idx)
    
    # Load logits for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Compute log_softmax
    # Find max
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + offsets
        col_mask = col_offsets < n_cols
        input_vals = tl.load(logits_ptr + row_start + col_offsets, mask=col_mask, other=float('-inf'))
        max_val = tl.maximum(max_val, input_vals)
    
    max_val = tl.max(max_val, axis=0)
    
    # Compute sum of exponentials
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + offsets
        col_mask = col_offsets < n_cols
        input_vals = tl.load(logits_ptr + row_start + col_offsets, mask=col_mask, other=0.0)
        exp_vals = tl.exp(input_vals - max_val)
        sum_exp += exp_vals
    
    sum_exp = tl.sum(sum_exp, axis=0)
    
    # Compute log_softmax
    log_probs = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + offsets
        col_mask = col_offsets < n_cols
        input_vals = tl.load(logits_ptr + row_start + col_offsets, mask=col_mask, other=0.0)
        log_prob_vals = input_vals - max_val - tl.log(sum_exp)
        tl.store(log_probs + col_offsets, log_prob_vals, mask=col_mask)
    
    # Extract the loss for the correct class
    correct_log_prob = tl.load(log_probs + target_idx)
    loss = -correct_log_prob
    
    # Store the result
    tl.store(output_ptr + row_idx, loss)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine the number of blocks needed
    grid = lambda meta: (n_rows,)
    
    # Launch the Triton kernel
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of cross entropy loss"""
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    # Prepare output tensor
    out = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    # Determine the number of blocks needed
    grid = lambda meta: (n_rows,)
    
    # Launch the Triton kernel
    cross_entropy_kernel[grid](logits, targets, out, n_rows, n_cols, BLOCK_SIZE=1024)
    
    # Return mean of losses
    return out.mean()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)