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
    
    # Calculate row start offset
    row_start = row_id * n_cols
    
    # Load input row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_id + 1) * n_cols
    input_row = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find max value in the row
    max_val = tl.max(input_row)
    
    # Compute exp(x - max_val) and sum
    exp_sum = 0.0
    exp_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for i in range(0, n_cols, BLOCK_SIZE):
        off = row_start + i + tl.arange(0, BLOCK_SIZE)
        msk = off < (row_id + 1) * n_cols
        x = tl.load(input_ptr + off, mask=msk, other=-float('inf'))
        exp_x = tl.exp(x - max_val)
        tl.store(output_ptr + off, exp_x, mask=msk)
        exp_sum += tl.sum(exp_x, axis=0)
    
    # Normalize to get log_softmax
    log_sum = tl.log(exp_sum)
    for i in range(0, n_cols, BLOCK_SIZE):
        off = row_start + i + tl.arange(0, BLOCK_SIZE)
        msk = off < (row_id + 1) * n_cols
        exp_val = tl.load(output_ptr + off, mask=msk, other=0.0)
        normalized = exp_val / exp_sum
        log_normalized = tl.log(normalized)
        tl.store(output_ptr + off, log_normalized, mask=msk)

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
    
    # Load target index
    target_idx = tl.load(targets_ptr + row_id)
    
    # Calculate row start offset
    row_start = row_id * n_cols
    
    # Load logits for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_id + 1) * n_cols
    logits_row = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find max value in the row for numerical stability
    max_val = tl.max(logits_row)
    
    # Compute log-sum-exp
    exp_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        off = row_start + i + tl.arange(0, BLOCK_SIZE)
        msk = off < (row_id + 1) * n_cols
        x = tl.load(logits_ptr + off, mask=msk, other=-float('inf'))
        exp_x = tl.exp(x - max_val)
        exp_sum += tl.sum(exp_x, axis=0)
    
    log_sum_exp = tl.log(exp_sum)
    
    # Get the log probability of the correct class
    correct_log_prob = logits_row[target_idx] - max_val - log_sum_exp
    
    # Store negative log likelihood loss
    tl.store(output_ptr + row_id, -correct_log_prob)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Choose block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_rows,)
    
    log_softmax_kernel[grid](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of cross entropy loss"""
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    output = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_rows,)
    
    cross_entropy_kernel[grid](logits, targets, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return torch.mean(output)

class ModelNew(nn.Module):
    """
    An optimized model using Triton kernels for computing Cross Entropy Loss.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use Triton kernel for cross entropy computation
        return triton_cross_entropy(predictions, targets)

# For compatibility with existing interface
def get_inputs():
    batch_size = 32768
    num_classes = 4096
    input_shape = (num_classes,)
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []