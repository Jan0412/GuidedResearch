import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    predictions_ptr,  # Pointer to predictions [batch_size, num_classes]
    targets_ptr,      # Pointer to targets [batch_size]
    output_ptr,       # Pointer to output scalar
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute per-sample loss and average
    # Each block processes one sample (row)
    sample_idx = tl.program_id(0)
    
    if sample_idx >= batch_size:
        return
    
    # Calculate starting offsets for this sample
    predictions_offset = sample_idx * num_classes
    
    # Load the current sample's predictions
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    
    # Load predictions for this sample
    sample_preds = tl.load(predictions_ptr + predictions_offset + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (online softmax)
    max_pred = tl.max(sample_preds, axis=0)
    
    # Compute exp(pred - max)
    exp_preds = tl.exp(sample_preds - max_pred)
    
    # Compute sum of exps
    sum_exp = tl.sum(exp_preds, axis=0)
    
    # Compute log sum exp = max + log(sum_exp)
    log_sum_exp = max_pred + tl.log(sum_exp)
    
    # Load target index for this sample
    target_idx = tl.load(targets_ptr + sample_idx)
    
    # Compute log probability of target class
    target_pred = tl.load(predictions_ptr + predictions_offset + target_idx)
    
    # Compute negative log likelihood: -(target_pred - log_sum_exp) = log_sum_exp - target_pred
    loss = log_sum_exp - target_pred
    
    # Store the loss for this sample
    tl.store(output_ptr + sample_idx, loss)


@triton.jit
def reduce_kernel(
    input_ptr,  # Pointer to input array
    output_ptr, # Pointer to output scalar
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Reduce array to single value (mean)
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Simple reduction - in practice might want hierarchical reduction
    sum_val = tl.sum(data, axis=0)
    count = tl.sum(mask.to(tl.float32), axis=0)
    
    if pid == 0:
        mean_val = sum_val / count
        tl.store(output_ptr, mean_val)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute cross entropy loss using Triton kernel.
    
    Args:
        predictions: [batch_size, num_classes] float32 tensor
        targets: [batch_size] int64 tensor with class indices
    
    Returns:
        scalar float32 tensor with mean cross entropy loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.dim() == 2, "Predictions must be 2D"
    assert targets.dim() == 1, "Targets must be 1D"
    assert predictions.size(0) == targets.size(0), "Batch size must match"
    assert targets.min() >= 0 and targets.max() < predictions.size(1), "Target indices out of range"
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.size(0)
    num_classes = predictions.size(1)
    
    # Allocate output for per-sample losses
    per_sample_losses = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
    
    # Launch kernel - one block per sample
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    cross_entropy_kernel[grid](
        predictions, targets, per_sample_losses,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reduce per-sample losses to scalar mean
    output = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
    reduce_grid = lambda meta: (1,) if batch_size > 0 else (0,)
    
    # For small batch sizes, we can do single-block reduction
    if batch_size <= BLOCK_SIZE:
        reduce_kernel[1](per_sample_losses, output, batch_size, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # For larger batches, use a more sophisticated reduction or fall back to torch.mean
        # But for simplicity, we'll use torch.mean here as the final reduction step
        return torch.mean(per_sample_losses)
    
    return output[0]


class ModelNew(nn.Module):
    """
    Optimized model that computes Cross Entropy Loss using Triton kernels.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our optimized Triton-based cross entropy
        return triton_cross_entropy(predictions, targets)