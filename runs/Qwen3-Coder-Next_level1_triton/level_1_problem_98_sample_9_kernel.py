import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kl_div_kernel(
    log_predictions_ptr,  # Pointer to log(predictions)
    targets_ptr,          # Pointer to targets (softmax output)
    out_ptr,              # Pointer to output (scalar result)
    n_elements,           # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load log(predictions) and targets
    log_p = tl.load(log_predictions_ptr + offsets, mask=mask, other=0.0)
    q = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence: q * (log(q) - log(p)) = q * log(q/p)
    # But since we have log(p), we compute: q * (log(q) - log(p))
    log_q = tl.log(q + 1e-12)  # Add small epsilon to avoid log(0)
    kl_values = q * (log_q - log_p)
    
    # Accumulate sum for batchmean reduction
    # We'll sum all elements and divide by batch_size at the end
    total_sum = tl.sum(kl_values, axis=0)
    
    # Store partial sum for each block (we'll do final reduction on CPU/GPU)
    # For simplicity, we'll use atomic add to accumulate to a single accumulator
    # But for better performance, we can use a two-pass approach
    # Here we use atomic add for simplicity
    tl.atomic_add(out_ptr, total_sum)


@triton.jit
def reduce_sum_kernel(
    input_ptr,    # Pointer to partial sums
    output_ptr,   # Pointer to final sum
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    
    tl.store(output_ptr, total)


def triton_kl_div(log_predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute KL divergence using Triton kernel.
    
    Args:
        log_predictions: log(predictions) where predictions are softmax outputs
        targets: target distribution (softmax outputs)
        
    Returns:
        KL divergence with batchmean reduction
    """
    assert log_predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    log_predictions = log_predictions.contiguous()
    targets = targets.contiguous()
    
    # Validate input shapes
    assert log_predictions.shape == targets.shape, "Input shapes must match"
    
    n_elements = log_predictions.numel()
    batch_size = log_predictions.shape[0]
    
    # For batchmean, we compute sum over all elements and divide by batch_size
    # We'll use a two-pass approach: first compute partial sums per block, then sum them
    
    # Use a reasonable block size
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Allocate buffer for partial sums
    num_blocks = grid({'BLOCK_SIZE': BLOCK_SIZE})[0]
    partial_sums = torch.zeros(num_blocks, device=log_predictions.device, dtype=log_predictions.dtype)
    
    # Launch kernel to compute partial sums
    kl_div_kernel[grid](log_predictions, targets, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Now reduce the partial sums
    final_sum = torch.zeros((), device=log_predictions.device, dtype=log_predictions.dtype)
    reduce_grid = lambda meta: ((num_blocks + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    reduce_sum_kernel[reduce_grid](partial_sums, final_sum, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    
    # Apply batchmean reduction: divide by batch_size
    return final_sum / batch_size


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Compute log(predictions) first
        log_predictions = torch.log(predictions + 1e-12)
        # Use Triton kernel for KL divergence computation
        return triton_kl_div(log_predictions, targets)