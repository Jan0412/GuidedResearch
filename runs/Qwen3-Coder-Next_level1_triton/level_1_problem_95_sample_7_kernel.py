import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    X_ptr,          # Input logits [batch_size, num_classes]
    Y_ptr,          # Target indices [batch_size]
    out_ptr,        # Output loss [batch_size] (for reduction later)
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one batch element
    batch_idx = tl.program_id(0)
    
    # Get pointer to current batch element
    X_batch_ptr = X_ptr + batch_idx * num_classes
    Y_batch_ptr = Y_ptr + batch_idx
    
    # Load target index
    target_idx = tl.load(Y_batch_ptr).to(tl.int32)
    
    # Compute max for numerical stability (online softmax)
    max_val = -float('inf')
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        vals = tl.load(X_batch_ptr + offsets, mask=mask, other=-float('inf'))
        batch_max = tl.max(vals)
        max_val = tl.maximum(max_val, batch_max)
    
    # Compute sum of exponentials
    sum_exp = 0.0
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        vals = tl.load(X_batch_ptr + offsets, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals * mask)
    
    # Compute log sum exp
    log_sum_exp = max_val + tl.log(sum_exp)
    
    # Get target logit
    target_val = tl.load(X_batch_ptr + target_idx)
    
    # Compute cross entropy loss for this sample: -target_logit + log_sum_exp
    loss = -target_val + log_sum_exp
    
    # Store result
    tl.store(out_ptr + batch_idx, loss)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute cross entropy loss using Triton kernel.
    
    Args:
        predictions: [batch_size, num_classes] logits
        targets: [batch_size] target class indices
    
    Returns:
        Scalar tensor with mean cross entropy loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.dim() == 2, "Predictions must be 2D"
    assert targets.dim() == 1, "Targets must be 1D"
    assert predictions.shape[0] == targets.shape[0], "Batch size must match"
    
    batch_size, num_classes = predictions.shape
    
    # Ensure contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Allocate output for per-sample losses
    per_sample_losses = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
    
    # Launch kernel - one program per batch element
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    cross_entropy_kernel[grid](
        predictions, targets, per_sample_losses,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean across batch
    return torch.mean(per_sample_losses)


class ModelNew(nn.Module):
    """
    Optimized model that computes Cross Entropy Loss using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)