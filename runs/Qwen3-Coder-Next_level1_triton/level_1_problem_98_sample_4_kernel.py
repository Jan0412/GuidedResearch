import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kl_divergence_kernel(
    predictions_ptr,  # Pointer to predictions (P)
    targets_ptr,      # Pointer to targets (Q)
    output_ptr,       # Pointer to output
    n_elements,       # Total number of elements
    stride,           # Stride between batches
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    # Compute starting offset for this batch
    batch_start = batch_idx * stride
    
    # Initialize accumulator for this batch
    sum_val = 0.0
    
    # Process in blocks
    for start in range(0, stride, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < stride
        
        # Load predictions and targets
        p = tl.load(predictions_ptr + batch_start + offsets, mask=mask, other=0.0)
        q = tl.load(targets_ptr + batch_start + offsets, mask=mask, other=0.0)
        
        # Compute KL divergence contribution: p * log(p/q) = p * (log(p) - log(q))
        # Use tl.where to avoid log(0) issues
        log_p = tl.log(p + 1e-12)
        log_q = tl.log(q + 1e-12)
        kl_contrib = p * (log_p - log_q)
        
        # Accumulate
        sum_val += tl.sum(kl_contrib * mask)
    
    # Store result for this batch
    tl.store(output_ptr + batch_idx, sum_val)


@triton.jit
def mean_reduction_kernel(
    batch_results_ptr,  # Pointer to batch-wise results
    output_ptr,         # Pointer to final output (scalar)
    n_batches,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global sum with block-level reduction
    total_sum = 0.0
    
    for start in range(0, n_batches, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_batches
        
        batch_val = tl.load(batch_results_ptr + offsets, mask=mask, other=0.0)
        total_sum += tl.sum(batch_val * mask)
    
    # Compute mean
    mean_val = total_sum / n_batches
    tl.store(output_ptr, mean_val)


def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute KL divergence with batchmean reduction using Triton kernels.
    
    Args:
        predictions: Tensor of shape (batch_size, dim) containing predicted probabilities
        targets: Tensor of shape (batch_size, dim) containing target probabilities
        
    Returns:
        Scalar tensor with mean KL divergence across batch
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Input tensors must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, dim = predictions.shape
    
    # Allocate output tensor for batch-wise results
    batch_results = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
    
    # First kernel: compute KL divergence for each batch
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    kl_divergence_kernel[grid](
        predictions,
        targets,
        batch_results,
        batch_size * dim,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: compute mean across batches
    output = torch.empty((), device=predictions.device, dtype=predictions.dtype)
    mean_reduction_kernel[(1,)](
        batch_results,
        output,
        batch_size,
        BLOCK_SIZE=256
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)