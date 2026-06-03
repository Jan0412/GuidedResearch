import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,  # Pointer to predictions
    targets_ptr,      # Pointer to targets
    output_ptr,       # Pointer to output (scalar)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Reduction variable for accumulating loss
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over the input in blocks
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load predictions and targets
        predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        # Compute difference
        diff = predictions - targets
        
        # Compute absolute value
        abs_diff = tl.abs(diff)
        
        # Compute smooth L1: 0.5 * x^2 if |x| < 1, else |x| - 0.5
        # Using boolean mask for conditional computation
        small_mask = abs_diff < 1.0
        
        # For small values: 0.5 * diff^2
        small_loss = 0.5 * diff * diff
        
        # For large values: |diff| - 0.5
        large_loss = abs_diff - 0.5
        
        # Combine using the mask
        loss = tl.where(small_mask, small_loss, large_loss)
        
        # Accumulate the loss
        acc += tl.sum(loss, axis=0)
    
    # Store the final accumulated loss (scaled by 1/n_elements for mean loss)
    # For the default reduction='mean', we need to divide by n_elements
    # But the original smooth_l1_loss uses reduction='mean' by default
    # So we'll store the mean loss
    mean_loss = acc / n_elements
    tl.store(output_ptr, mean_loss)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute Smooth L1 (Huber) loss using Triton kernel.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        Smooth L1 loss (scalar tensor)
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor (scalar)
    out = torch.empty(1, device=predictions.device, dtype=predictions.dtype)
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Launch the Triton kernel with grid of 1 block since we're doing reduction
    # But Triton reduction kernels need special handling for the output
    # We'll use a simpler approach with a single block for small-to-medium tensors
    # For larger tensors, we might need grid-stride loop or multiple reductions
    
    # For the reduction, we'll use a single block that handles all elements
    # via grid-stride loop in the kernel
    grid = (1,)
    
    smooth_l1_loss_kernel[grid](
        predictions, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that computes Smooth L1 (Huber) Loss using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use our custom Triton implementation instead of PyTorch's
        return triton_smooth_l1_loss(predictions, targets)