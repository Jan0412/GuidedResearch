import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def kl_divergence_kernel(
    log_pred_ptr,  # Pointer to log(predictions)
    target_ptr,    # Pointer to targets
    output_ptr,    # Pointer to output
    n_elements,    # Total number of elements
    stride,        # Stride between batches
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a batch
    batch_idx = tl.program_id(0)
    
    # Compute starting offset for this batch
    offset = batch_idx * stride
    
    # Create offsets for this batch's elements
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_idx + 1) * stride
    
    # Load log(predictions) and targets
    log_pred = tl.load(log_pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute log(predictions) - log(targets) = log(predictions/targets)
    # But we need target * log(predictions/targets)
    # To avoid log(0), we mask out zero targets
    nonzero_mask = target > 0.0
    safe_target = tl.where(nonzero_mask, target, 0.0)
    safe_log_pred = tl.where(nonzero_mask, log_pred, 0.0)
    
    # Compute target * (log(target) - log(predictions))
    # But kl_div expects log(predictions) as first argument, so:
    # KL = target * (log(target) - log(predictions))
    # Actually: KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    # Here target is P, predictions is Q, but we have log(predictions)
    # So: KL = sum(target * (log(target) - log(predictions)))
    
    # However, log(target) when target=0 is problematic, but target is softmax so it should be > 0
    # But to be safe, we only compute where target > 0
    
    # Compute log(target) - log(predictions) = log(target/predictions)
    # Then multiply by target
    log_target = tl.where(nonzero_mask, tl.log(target), 0.0)
    kl_element = safe_target * (log_target - safe_log_pred)
    
    # Sum over the batch dimension
    sum_kl = tl.sum(kl_element, axis=0)
    
    # Store the sum for this batch
    tl.store(output_ptr + batch_idx, sum_kl)


@triton.jit
def reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel reduces multiple batch sums to a single mean
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(input_ptr + offset, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    
    # Store the total sum
    tl.store(output_ptr, total)


def triton_kl_div(log_pred: torch.Tensor, target: torch.Tensor):
    """
    Triton implementation of KL divergence with batchmean reduction.
    
    Args:
        log_pred: log(predictions) tensor
        target: targets tensor (should be normalized probabilities)
        
    Returns:
        Scalar tensor with mean KL divergence across batch
    """
    assert log_pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    log_pred = log_pred.contiguous()
    target = target.contiguous()
    
    batch_size = log_pred.size(0)
    seq_len = log_pred.size(1) if log_pred.dim() > 1 else 1
    
    # For 1D case (single element per batch)
    if log_pred.dim() == 1:
        # Reshape to 2D for consistent handling
        log_pred = log_pred.unsqueeze(1)
        target = target.unsqueeze(1)
        seq_len = 1
    
    # Prepare output for batch sums
    batch_sums = torch.empty(batch_size, device=log_pred.device, dtype=log_pred.dtype)
    
    # Determine block size (round up to multiple of 32)
    BLOCK_SIZE = 128
    grid = (batch_size,)
    
    # Launch KL divergence kernel for each batch
    kl_divergence_kernel[grid](
        log_pred, target, batch_sums,
        log_pred.numel(),
        log_pred.size(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Now reduce batch sums to get mean
    total_sum = torch.empty(1, device=log_pred.device, dtype=log_pred.dtype)
    reduction_kernel[1](
        batch_sums, total_sum,
        batch_size,
        BLOCK_SIZE=128
    )
    
    # Compute mean
    return total_sum[0] / batch_size


class ModelNew(nn.Module):
    """
    Optimized model that computes Kullback-Leibler Divergence using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Compute log of predictions (since predictions are already softmax, just log)
        log_predictions = torch.log(predictions)
        # Use our optimized Triton kernel for KL divergence
        return triton_kl_div(log_predictions, targets)