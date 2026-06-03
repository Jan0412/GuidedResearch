import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_kernel(
    predictions_ptr,  # Pointer to predictions
    targets_ptr,      # Pointer to targets
    output_ptr,       # Pointer to output (scalar MSE)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared error
    diff = predictions - targets
    squared_error = diff * diff
    
    # Use atomic add to accumulate into a shared accumulator
    # Each block accumulates its partial sum
    sum = tl.sum(squared_error, axis=0)
    
    # Store partial sum to a buffer (we'll reduce these partial sums later)
    # For simplicity, we'll use a global accumulator approach with atomics
    tl.atomic_add(output_ptr, sum)


def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute Mean Squared Error using Triton kernel.
    
    Parameters:
        predictions: Predicted values tensor
        targets: Ground truth values tensor
    
    Returns:
        Scalar MSE loss
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    
    # Grid size: number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Create output tensor for partial sums (one per block)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=predictions.device)
    
    # Launch kernel to compute partial sums
    mse_kernel[grid](predictions, targets, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Final reduction on GPU (this is small, so PyTorch is fine)
    total_sum = torch.sum(partial_sums)
    mse = total_sum / n_elements
    
    return mse


class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss using Triton kernels.
    
    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)