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
    GROUP_SIZE_M: tl.constexpr
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Each program processes one row
    if row_idx >= n_rows:
        return
    
    # Calculate the starting position for this row
    row_start = row_idx * n_cols
    
    # Initialize max_val and sum_exp
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process the row in chunks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load input values
        inp_vals = tl.load(inp_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        
        # Compute max_val for this chunk
        chunk_max = tl.max(inp_vals, axis=0)
        max_val = tl.maximum(max_val, chunk_max)
    
    # Broadcast max_val to all threads in the block
    max_val = tl.broadcast_to(max_val, [BLOCK_SIZE])
    
    # Compute exp(x - max_x) and sum
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        inp_vals = tl.load(inp_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Compute exp(x - max_x)
        exp_vals = tl.exp(inp_vals - max_val)
        
        # Accumulate sum
        sum_exp += exp_vals
        
        # Store the result
        tl.store(out_ptr + row_start + offsets, exp_vals, mask=mask)
    
    # Normalize by sum
    sum_exp = tl.sum(sum_exp, axis=0)
    sum_exp = tl.broadcast_to(sum_exp, [BLOCK_SIZE])
    
    # Normalize the exponentials
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        exp_vals = tl.load(out_ptr + row_start + offsets, mask=mask, other=0.0)
        normalized = exp_vals / sum_exp
        tl.store(out_ptr + row_start + offsets, normalized, mask=mask)

@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Get target label for this row
    target = tl.load(targets_ptr + row_idx)
    
    # Calculate row start
    row_start = row_idx * n_cols
    
    # Compute log_softmax
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # First pass: find max
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        chunk_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, chunk_max)
    
    # Broadcast max value
    max_val = tl.broadcast_to(max_val, [BLOCK_SIZE])
    
    # Second pass: compute sum_exp
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += exp_vals
    
    # Broadcast sum
    sum_exp = tl.sum(sum_exp, axis=0)
    sum_exp = tl.broadcast_to(sum_exp, [BLOCK_SIZE])
    
    # Third pass: compute log_softmax and extract target logit
    target_logit = tl.full([1], 0.0, dtype=tl.float32)
    
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        vals = tl.load(logits_ptr + row_start + offsets, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        log_softmax_vals = vals - max_val - tl.log(sum_exp)
        
        # Extract target logit
        if target < n_cols:
            target_mask = offsets == target
            target_logit = tl.where(target_mask, log_softmax_vals, target_logit)
        
        # Store log_softmax values
        tl.store(out_ptr + row_start + offsets, log_softmax_vals, mask=mask)
    
    # Store negative log likelihood
    loss = -target_logit
    tl.store(out_ptr + n_rows * n_cols + row_idx, loss)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Configure block size
    BLOCK_SIZE = 1024
    GRID_SIZE = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    log_softmax_kernel[GRID_SIZE](
        x, 
        out,
        n_cols,
        n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE_M=8
    )
    
    return out

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of cross entropy loss"""
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    
    # Allocate output tensor (for log_softmax results + losses)
    out = torch.empty(n_rows, n_cols + 1, dtype=torch.float32, device=logits.device)
    
    # Configure block size
    BLOCK_SIZE = 1024
    GRID_SIZE = (n_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    cross_entropy_kernel[GRID_SIZE](
        logits,
        targets,
        out,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Extract the loss values
    loss = out[:, -1].mean()
    return loss

class ModelNew(nn.Module):
    """
    An optimized version of the model using Triton kernels for cross entropy computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our custom Triton implementation
        return triton_cross_entropy(predictions, targets)