import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    X_ptr,  # Input predictions [batch_size, num_classes]
    Y_ptr,  # Target indices [batch_size]
    loss_ptr,  # Output loss [batch_size]
    N_CLASSES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Offset to the start of this batch's data
    x_offset = batch_idx * N_CLASSES
    
    # Load the targets (convert to pointer arithmetic)
    target = tl.load(Y_ptr + batch_idx)
    
    # Load all logits for this sample
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_CLASSES
    
    # Load logits
    logits = tl.load(X_ptr + x_offset + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (online softmax)
    max_logit = tl.max(logits, axis=0)
    
    # Compute exponentials with numerical stability
    exp_logits = tl.exp(logits - max_logit)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_logits, axis=0)
    
    # Compute log(sum(exp)) = max_logit + log(sum_exp)
    log_sum_exp = max_logit + tl.log(sum_exp)
    
    # Get the target logit
    target_logit = tl.load(X_ptr + x_offset + target)
    
    # Cross entropy loss = -(target_logit - log_sum_exp) = log_sum_exp - target_logit
    loss = log_sum_exp - target_logit
    
    # Store the result
    tl.store(loss_ptr + batch_idx, loss)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute cross entropy loss using Triton kernel.
    
    Args:
        predictions: [batch_size, num_classes] tensor of logits
        targets: [batch_size] tensor of target class indices
    
    Returns:
        Scalar tensor with mean cross entropy loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.dim() == 2, "Predictions must be 2D"
    assert targets.dim() == 1, "Targets must be 1D"
    assert predictions.shape[0] == targets.shape[0], "Batch size mismatch"
    
    batch_size, num_classes = predictions.shape
    
    # Ensure contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Allocate output for per-sample losses
    losses = torch.empty(batch_size, dtype=predictions.dtype, device=predictions.device)
    
    # Set block size - tuned for typical GPU architectures
    BLOCK_SIZE = min(1024, triton.next_power_of_2(num_classes))
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    cross_entropy_kernel[grid](
        predictions, 
        targets, 
        losses, 
        N_CLASSES=num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return mean loss (match PyTorch's default behavior)
    return losses.mean()


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for cross entropy loss computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace PyTorch's cross_entropy with our Triton implementation
        return triton_cross_entropy(predictions, targets)