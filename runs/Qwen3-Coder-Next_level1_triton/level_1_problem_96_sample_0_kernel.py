import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output scalar
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Accumulate loss in local memory
    accumulator = tl.zeros((1,), dtype=tl.float32)
    
    # Calculate grid offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute residuals
    diff = predictions - targets
    
    # Smooth L1 loss: 
    # |x|^2 / 2 if |x| < 1
    # |x| - 0.5 otherwise
    abs_diff = tl.abs(diff)
    condition = abs_diff < 1.0
    
    # Compute loss per element
    loss = tl.where(
        condition,
        0.5 * diff * diff,
        abs_diff - 0.5
    )
    
    # Accumulate results
    accumulator += tl.sum(loss, axis=0, keep_dims=True)
    
    # Store partial sum to global memory
    tl.atomic_add(output_ptr, accumulator)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton implementation of Smooth L1 (Huber) Loss.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Ensure same shape
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    
    # Get total number of elements
    n_elements = predictions.numel()
    
    # Prepare output tensor (scalar)
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    smooth_l1_loss_kernel[grid](
        predictions, targets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Average over number of elements to match PyTorch's default reduction='mean'
    # Note: PyTorch's smooth_l1_loss uses reduction='mean' by default, 
    # so we divide by n_elements to get the mean
    return output / n_elements


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Smooth L1 (Huber) Loss.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)