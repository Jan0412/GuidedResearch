import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load prediction and target values
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared difference
    diff = pred - target
    squared_diff = diff * diff
    
    # Store intermediate squared differences
    tl.store(out_ptr + offsets, squared_diff, mask=mask)

@triton.jit
def sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce sum within block
    local_sum = tl.sum(input_vals, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + tl.program_id(0), local_sum, mask=tl.program_id(0) < (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute MSE using Triton kernels
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Get total number of elements
    n_elements = predictions.numel()
    
    # First kernel: compute squared differences
    squared_diffs = torch.empty_like(predictions)
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    mse_kernel[grid](predictions, targets, squared_diffs, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: compute mean of squared differences
    # For this, we need to reduce the squared differences to a scalar
    # We'll do this in two steps: partial sums then final reduction
    
    # Compute partial sums
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=predictions.device)
    
    sum_kernel[grid](squared_diffs, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Final reduction on CPU or use a simple approach
    # Since we're computing mean, we divide by n_elements at the end
    total_sum = partial_sums.sum()
    mean = total_sum / n_elements
    
    return mean

class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.
    Optimized with Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)