import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output scalar
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask)
    targets = tl.load(targets_ptr + offsets, mask=mask)
    
    # Compute squared difference
    diff = predictions - targets
    squared_diff = diff * diff
    
    # Use atomic add to accumulate into global sum
    tl.atomic_add(output_ptr, tl.sum(squared_diff))


def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute Mean Squared Error loss using Triton kernel.
    
    Args:
        predictions: Predicted tensor
        targets: Target tensor
    Returns:
        Mean squared error loss (scalar)
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Get total number of elements
    n_elements = predictions.numel()
    
    # Prepare output tensor (scalar)
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Configure kernel
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    mse_loss_kernel[grid](predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean by dividing by number of elements
    return output[0] / n_elements


class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss for regression tasks
    using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse_loss(predictions, targets)