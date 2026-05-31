import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor (same shape as predictions)
    out_ptr,          # Pointer to output scalar (mean hinge loss)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Accumulator for reduction
    sum_ = tl.zeros((1,), dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load predictions and targets
        preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        targs = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        # Compute hinge loss: max(0, 1 - pred * target)
        margin = 1.0 - preds * targs
        loss = tl.where(margin > 0, margin, 0.0)
        
        # Accumulate sum
        sum_ += tl.sum(loss, axis=0)
    
    # Write result as mean (sum / n_elements)
    mean_loss = sum_ / n_elements
    tl.store(out_ptr, mean_loss)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes hinge loss using Triton kernel: mean(max(0, 1 - pred * target))
    
    Args:
        predictions: Tensor of shape (batch_size, *input_shape)
        targets: Tensor of shape (batch_size, *input_shape) with values in {-1, 1}
        
    Returns:
        Scalar tensor containing mean hinge loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape."
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    # Flatten inputs for kernel processing
    n_elements = predictions.numel()
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)

    # Output tensor (scalar)
    out = torch.empty(1, device=predictions.device, dtype=predictions.dtype)

    # Tune block size
    BLOCK_SIZE = 1024

    # Launch kernel with single block (since output is scalar, we do reduction in-kernel)
    grid = (1,)
    hinge_loss_kernel[grid](predictions_flat, targets_flat, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification tasks.
    Uses custom Triton kernel instead of PyTorch operations.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Replace PyTorch operations with Triton kernel
        return triton_hinge_loss(predictions, targets)