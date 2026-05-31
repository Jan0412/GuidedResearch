import torch
import torch.nn as nn
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
    
    # Calculate row offset
    row_offset = row_id * n_cols
    input_row = input_ptr + row_offset
    output_row = output_ptr + row_offset
    
    # Find max value in the row for numerical stability
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_row + cols, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(vals))
    
    # Compute sum of exponentials
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_row + cols, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals)
    
    # Compute log(sum_exp) + max_val
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Compute final log_softmax values
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_row + cols, mask=mask, other=0.0)
        result = vals - log_sum_exp
        tl.store(output_row + cols, result, mask=mask)

@triton.jit
def kl_div_kernel(
    log_predictions_ptr,
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
    
    # Calculate row offsets
    row_offset = row_id * n_cols
    log_pred_row = log_predictions_ptr + row_offset
    target_row = targets_ptr + row_offset
    output_row = output_ptr + row_offset
    
    # Compute KL divergence for this row
    kl_sum = tl.zeros([1], dtype=tl.float32)
    total_elements = 0
    
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        log_pred_vals = tl.load(log_pred_row + cols, mask=mask, other=0.0)
        target_vals = tl.load(target_row + cols, mask=mask, other=0.0)
        
        # Compute kl divergence contribution: target * (log_target - log_pred)
        kl_contrib = target_vals * (tl.log(target_vals + 1e-8) - log_pred_vals)
        kl_sum += tl.sum(kl_contrib)
        total_elements += tl.sum(tl.where(mask, 1, 0))
    
    # Store average for this row
    avg_kl = kl_sum / total_elements
    tl.store(output_row, avg_kl)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (math.ceil(n_rows),)
    
    log_softmax_kernel[grid](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output

def triton_kl_div(log_predictions: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of KL divergence"""
    assert log_predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    log_predictions = log_predictions.contiguous()
    targets = targets.contiguous()
    
    n_rows, n_cols = log_predictions.shape
    output = torch.empty(n_rows, dtype=torch.float32, device=log_predictions.device)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (math.ceil(n_rows),)
    
    kl_div_kernel[grid](log_predictions, targets, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    # Return mean over all rows
    return output.mean()

class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.
    Optimized with custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use Triton implementation instead of PyTorch native functions
        log_predictions = triton_log_softmax(predictions)
        return triton_kl_div(log_predictions, targets)