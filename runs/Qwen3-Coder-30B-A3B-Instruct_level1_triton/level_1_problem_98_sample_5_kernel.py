import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate the starting position for this batch and sequence
    start_pos = batch_idx * seq_len + seq_idx * seq_len
    
    # Each program processes one element in the batch
    block_start = tl.program_id(2) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load prediction and target values
    pred = tl.load(predictions_ptr + start_pos + offsets, mask=mask, other=0.0)
    target = tl.load(targets_ptr + start_pos + offsets, mask=mask, other=0.0)
    
    # Compute KL divergence: sum(target * log(target / prediction))
    # We compute log(target / prediction) = log(target) - log(prediction)
    # For numerical stability, we use the identity: 
    # target * log(target / prediction) = target * log(target) - target * log(prediction)
    
    # Handle edge cases where prediction or target might be zero
    pred_safe = tl.where(pred > 0.0, pred, 1e-8)
    target_safe = tl.where(target > 0.0, target, 1e-8)
    
    # Compute log values
    log_pred = tl.log(pred_safe)
    log_target = tl.log(target_safe)
    
    # Compute the KL divergence contribution
    kl_contribution = target * (log_target - log_pred)
    
    # Sum up contributions for this batch/sequence
    kl_sum = tl.sum(kl_contribution, axis=0)
    
    # Store the result for this batch/sequence
    if block_start == 0:
        tl.store(output_ptr + batch_idx * seq_len + seq_idx, kl_sum, mask=True)

# Optimized version that does both log and kl_div computation in one kernel
@triton.jit
def log_kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting position for this batch
    start_pos = batch_idx * seq_len
    
    # Each program processes a chunk of the sequence
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load prediction and target values
    pred = tl.load(predictions_ptr + start_pos + offsets, mask=mask, other=0.0)
    target = tl.load(targets_ptr + start_pos + offsets, mask=mask, other=0.0)
    
    # Compute log(predictions) and handle numerical stability
    pred_safe = tl.where(pred > 0.0, pred, 1e-8)
    log_pred = tl.log(pred_safe)
    
    # Compute log(targets) and handle numerical stability
    target_safe = tl.where(target > 0.0, target, 1e-8)
    log_target = tl.log(target_safe)
    
    # Compute the KL divergence contribution
    kl_contribution = target * (log_target - log_pred)
    
    # Compute the sum across the sequence dimension
    kl_sum = tl.sum(kl_contribution, axis=0)
    
    # Store the result for this batch
    tl.store(output_ptr + batch_idx, kl_sum, mask=True)

def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute KL divergence using a Triton kernel.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, seq_len = predictions.shape
    
    # Prepare output tensor
    out = torch.empty(batch_size, dtype=torch.float32, device=predictions.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (batch_size, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Launch the kernel
    log_kl_div_kernel[grid](
        predictions,
        targets,
        out,
        predictions.numel(),
        batch_size,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean over all batches
    return out.mean()

class ModelNew(nn.Module):
    """
    An optimized model that computes Kullback-Leibler Divergence using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Apply log transformation followed by KL divergence computation
        return triton_kl_div(predictions, targets)