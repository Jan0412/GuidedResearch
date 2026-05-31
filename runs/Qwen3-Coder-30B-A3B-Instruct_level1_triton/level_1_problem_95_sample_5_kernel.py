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
    
    # Calculate row start
    row_start = row_idx * n_cols
    
    # Load input row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * n_cols
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Find max value for numerical stability
    max_val = tl.max(input_vals)
    
    # Compute exp(x - max_val)
    exp_vals = tl.exp(input_vals - max_val)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_vals)
    
    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)
    
    # Compute final log_softmax: x - max - log(sum_exp)
    final_vals = input_vals - max_val - log_sum_exp
    
    # Store results
    tl.store(output_ptr + offsets, final_vals, mask=mask)

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
    
    # Calculate row start
    row_start = row_idx * n_cols
    
    # Load logits for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * n_cols
    
    # Load logits
    logits = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Load target class
    target = tl.load(targets_ptr + row_idx)
    
    # Find max for numerical stability
    max_val = tl.max(logits)
    
    # Compute exp(x - max_val)
    exp_vals = tl.exp(logits - max_val)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_vals)
    
    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)
    
    # Compute loss: -log(exp(logits[target]) / sum_exp)
    # = -(logits[target] - max_val - log(sum_exp))
    # = -logits[target] + max_val + log(sum_exp)
    target_logits = logits[target]
    loss = -target_logits + max_val + log_sum_exp
    
    # Store result
    tl.store(output_ptr + row_idx, loss)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Grid configuration
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    log_softmax_kernel[grid](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of cross entropy loss"""
    assert logits.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = logits.shape
    BLOCK_SIZE = 1024
    
    # Allocate output
    output = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    
    # Grid configuration
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    cross_entropy_kernel[grid](logits, targets, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return mean loss
    return output.mean()

class ModelNew(nn.Module):
    """
    An optimized model that computes Cross Entropy Loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use Triton kernel for cross entropy computation
        return triton_cross_entropy(predictions, targets)

# Input generation functions remain unchanged
batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []