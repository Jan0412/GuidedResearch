import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global position
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute difference
    diff = predictions - targets
    abs_diff = tl.abs(diff)
    
    # Smooth L1 loss: 0.5 * x^2 if |x| < 1, else |x| - 0.5
    loss = tl.where(
        abs_diff < 1.0,
        0.5 * diff * diff,
        abs_diff - 0.5
    )
    
    # Store intermediate result (we'll do reduction in host code for simplicity)
    tl.store(output_ptr + offsets, loss, mask=mask)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute Smooth L1 (Huber) loss using Triton kernel.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        
    Returns:
        Scalar tensor containing the mean loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Ensure same shape
    assert predictions.shape == targets.shape, "Input shapes must match"
    
    # Flatten for kernel processing
    n_elements = predictions.numel()
    
    # Create output buffer for element-wise losses
    element_losses = torch.empty_like(predictions)
    
    # Configure kernel launch
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    smooth_l1_loss_kernel[grid](
        predictions,
        targets,
        element_losses,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute mean loss (this is a simple reduction that's efficient on modern GPUs)
    return element_losses.mean()


class ModelNew(nn.Module):
    """
    Optimized model that computes Smooth L1 (Huber) Loss using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)