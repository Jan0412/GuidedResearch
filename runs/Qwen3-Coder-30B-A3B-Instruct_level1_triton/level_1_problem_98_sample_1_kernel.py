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
    sum_exp = tl.sum(exp_x, axis=0)
    log_sum_exp = tl.log(sum_exp)
    output = x_shifted - log_sum_exp
    
    tl.store(output_ptr + offsets, output, mask=mask)

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
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    log_p = tl.load(log_predictions_ptr + offsets, mask=mask, other=0.0)
    q = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence: sum(q * (log(q) - log(p)))
    # For each element, compute q * log(q/p) = q * (log(q) - log(p))
    kl = q * (tl.log(q + 1e-8) - log_p)  # Add small epsilon to avoid log(0)
    
    # Sum over all elements and divide by batch size for batchmean
    # Note: We'll compute the final reduction in Python
    tl.store(output_ptr + offsets, kl, mask=mask)

def triton_log_softmax(x: torch.Tensor):
    """Compute log-softmax using Triton kernel"""
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    log_softmax_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_kl_div(log_predictions: torch.Tensor, targets: torch.Tensor):
    """Compute KL divergence using Triton kernel"""
    assert log_predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    log_predictions = log_predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(log_predictions)
    
    # Number of elements in the tensor
    n_elements = log_predictions.numel()
    batch_size = log_predictions.shape[0]
    seq_len = log_predictions.shape[1] if len(log_predictions.shape) > 1 else 1
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    kl_div_kernel[grid](log_predictions, targets, out, n_elements, batch_size, seq_len, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reduce to batchmean
    result = out.sum() / batch_size
    return result

class ModelNew(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.
    Optimized with custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use Triton kernel for log operation instead of PyTorch's
        log_predictions = triton_log_softmax(predictions)
        
        # Use Triton kernel for KL divergence computation
        return triton_kl_div(log_predictions, targets)