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
    BLOCK_SIZE: tl.constexpr,
    NUM_CLASSES: tl.constexpr,
):
    # Each program processes one batch element
    batch_idx = tl.program_id(0)
    
    # Pointer to the start of this batch element's row
    x_offset = batch_idx * NUM_CLASSES
    
    # Load the target class index for this batch element
    target_idx = tl.load(Y_ptr + batch_idx)
    
    # Load all predictions for this batch element
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_CLASSES
    x_row = tl.load(X_ptr + x_offset + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (online softmax)
    x_max = tl.max(x_row, axis=0)
    
    # Compute exponentials with numerical stability
    x_exp = tl.exp(x_row - x_max)
    
    # Compute sum of exponentials
    x_sum = tl.sum(x_exp, axis=0)
    
    # Compute log sum exp (logsumexp)
    log_sum_exp = x_max + tl.log(x_sum)
    
    # Get the prediction for the target class
    target_val = tl.load(X_ptr + x_offset + target_idx)
    
    # Compute cross entropy loss: log_sum_exp - target_val
    loss = log_sum_exp - target_val
    
    # Store the loss for this batch element
    tl.store(loss_ptr + batch_idx, loss)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes cross entropy loss using Triton kernel.
    
    Args:
        predictions: Tensor of shape [batch_size, num_classes]
        targets: Tensor of shape [batch_size] with class indices
    
    Returns:
        Scalar tensor with mean cross entropy loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.dim() == 2, "Predictions must be 2D tensor."
    assert targets.dim() == 1, "Targets must be 1D tensor."
    assert predictions.size(0) == targets.size(0), "Batch size must match."
    
    batch_size, num_classes = predictions.shape
    
    # Ensure inputs are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Allocate output tensor for per-batch losses
    losses = torch.empty(batch_size, dtype=predictions.dtype, device=predictions.device)
    
    # Set block size
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Grid: one program per batch element
    grid = (batch_size,)
    
    # Launch the Triton kernel
    cross_entropy_kernel[grid](
        predictions, 
        targets, 
        losses,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_CLASSES=num_classes,
    )
    
    # Return mean loss
    return losses.mean()


class ModelNew(nn.Module):
    """
    Optimized model that computes Cross Entropy Loss using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)