import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes the mean squared error loss in a fused kernel.
    Each block accumulates a partial sum, then the final result is computed.
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference
    diff = predictions - targets
    squared_diff = diff * diff
    
    # Accumulate sum
    sum_val = tl.sum(squared_diff, axis=0)
    
    # Store partial sum for this block
    tl.atomic_add(output_ptr, sum_val)


def triton_mse_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes mean squared error loss using Triton kernel.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    
    # We need a single-element tensor to accumulate the sum
    # Use float32 for accumulation to match input precision
    partial_sum = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Grid of 1 block per required block of data
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the kernel
    mse_loss_kernel[grid](predictions, targets, partial_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean by dividing by number of elements
    return partial_sum[0] / n_elements


class ModelNew(nn.Module):
    """
    Optimized model that computes the Mean Squared Error loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse_loss(predictions, targets)