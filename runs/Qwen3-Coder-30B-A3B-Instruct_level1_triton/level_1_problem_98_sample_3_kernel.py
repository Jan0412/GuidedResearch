import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def log_softmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute log-softmax: log(exp(x_i) / sum(exp(x_j)))
    # First compute max for numerical stability
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp_x = tl.sum(exp_x, axis=0)
    log_sum_exp_x = tl.log(sum_exp_x)
    
    # Final result: x_i - max - log(sum(exp(x_j)))
    result = x_shifted - log_sum_exp_x
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def kl_div_kernel(
    log_predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    log_p = tl.load(log_predictions_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence: sum(t * (log(t) - log(p)))
    # For numerical stability, we use the identity:
    # t * (log(t) - log(p)) = t * log(t/p)
    # But when t=0, this becomes 0, so we handle that case
    kl = tl.where(t > 0, t * (tl.log(t) - log_p), 0.0)
    tl.store(output_ptr + offsets, kl, mask=mask)

@triton.jit
def reduce_mean_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Reduce along sequence dimension
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # For each batch item, sum all elements and divide by seq_len
    batch_offsets = offsets * seq_len
    total = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for i in range(seq_len):
        elem_offset = batch_offsets + i
        val = tl.load(input_ptr + elem_offset, mask=(elem_offset < n_elements), other=0.0)
        total += val
    
    mean_val = total / seq_len
    tl.store(output_ptr + offsets, mean_val, mask=mask)

def triton_log_softmax(x: torch.Tensor):
    """Triton implementation of log_softmax"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    log_softmax_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_kl_div(log_predictions: torch.Tensor, targets: torch.Tensor):
    """Triton implementation of KL divergence computation"""
    assert log_predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    log_predictions = log_predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(log_predictions)
    
    # Number of elements in the tensor
    n_elements = log_predictions.numel()
    BLOCK_SIZE = 1024
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    kl_div_kernel[grid](log_predictions, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_reduce_mean(x: torch.Tensor, batch_size: int, seq_len: int):
    """Triton implementation of reducing to mean across sequence dimension"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Determine the number of blocks needed
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    reduce_mean_kernel[grid](x, out, n_elements, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for KL divergence computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Apply log to predictions
        log_predictions = triton_log_softmax(predictions)
        
        # Compute KL divergence
        kl_div = triton_kl_div(log_predictions, targets)
        
        # Compute mean over batch
        batch_size = predictions.shape[0]
        seq_len = predictions.shape[1]
        result = triton_reduce_mean(kl_div, batch_size, seq_len)
        
        # Return the mean of the batch
        return result.mean()