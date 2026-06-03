import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    X_ptr,          # Pointer to input logits [batch_size, num_classes]
    Y_ptr,          # Pointer to targets [batch_size]
    out_ptr,        # Pointer to output loss [1]
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
    NUM_CLASSES_BLOCK: tl.constexpr,
):
    # Each program processes one batch element
    batch_idx = tl.program_id(0)
    
    # Load the target class index for this batch
    target_idx = tl.load(Y_ptr + batch_idx)
    
    # Compute max value for numerical stability (online softmax)
    x_start = batch_idx * num_classes
    max_val = -float('inf')
    for start in range(0, num_classes, NUM_CLASSES_BLOCK):
        offsets = start + tl.arange(0, NUM_CLASSES_BLOCK)
        mask = offsets < num_classes
        x_block = tl.load(X_ptr + x_start + offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(x_block, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Compute softmax denominator with max subtracted
    sum_exp = 0.0
    for start in range(0, num_classes, NUM_CLASSES_BLOCK):
        offsets = start + tl.arange(0, NUM_CLASSES_BLOCK)
        mask = offsets < num_classes
        x_block = tl.load(X_ptr + x_start + offsets, mask=mask, other=-float('inf'))
        x_shifted = x_block - max_val
        sum_exp += tl.sum(tl.exp(x_shifted), axis=0)
    
    # Compute log sum exp
    log_sum_exp = max_val + tl.log(sum_exp)
    
    # Get the target class logit
    target_val = tl.load(X_ptr + x_start + target_idx)
    
    # Cross entropy loss: -(target_logit - log_sum_exp) = log_sum_exp - target_logit
    loss = log_sum_exp - target_val
    
    # Store the loss
    tl.store(out_ptr + batch_idx, loss)


@triton.jit
def mean_kernel(
    x_ptr,          # Pointer to input [batch_size]
    out_ptr,        # Pointer to output [1]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute sum
    sum_val = 0.0
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x_block, axis=0)
    
    # Compute mean
    mean_val = sum_val / n_elements
    tl.store(out_ptr, mean_val)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute cross entropy loss using Triton kernels.
    
    Args:
        predictions: [batch_size, num_classes] tensor of logits
        targets: [batch_size] tensor of class indices
        
    Returns:
        scalar tensor with mean cross entropy loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.dim() == 2, "Predictions must be 2D"
    assert targets.dim() == 1, "Targets must be 1D"
    assert predictions.size(0) == targets.size(0), "Batch size must match"
    
    batch_size, num_classes = predictions.shape
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Allocate output for per-sample losses
    per_sample_losses = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)
    
    # Set up kernel parameters
    BLOCK_SIZE = 128
    NUM_CLASSES_BLOCK = 256  # Tunable parameter for processing classes
    
    # Grid: one program per batch element
    grid = (batch_size,)
    
    # Launch cross entropy kernel
    cross_entropy_kernel[grid](
        predictions, targets, per_sample_losses,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_CLASSES_BLOCK=NUM_CLASSES_BLOCK,
    )
    
    # Compute mean of per-sample losses
    total_loss = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
    mean_kernel[(1,)](
        per_sample_losses, total_loss, batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return total_loss


class ModelNew(nn.Module):
    """
    Optimized model that computes Cross Entropy Loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)