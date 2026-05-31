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
    
    # Load inputs
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared error: (pred - target) ** 2
    diff = pred - target
    squared_error = diff * diff
    
    # Store the squared error
    tl.store(out_ptr + offsets, squared_error, mask=mask)

@triton.jit
def mean_kernel(
    input_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum using atomic operations for parallel reduction
    # This is a simplified version - in practice you'd want proper reduction
    # But for this case we'll compute mean directly
    tl.store(out_ptr, tl.sum(x, axis=0) / n_elements, mask=True)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """Compute MSE with Triton kernel"""
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor for squared errors
    squared_errors = torch.empty_like(predictions)
    
    # Number of elements
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel to compute squared errors
    mse_kernel[grid](predictions, targets, squared_errors, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Now compute mean of squared errors
    # For this we can use a simpler approach since we're just reducing one value
    mean_result = torch.mean(squared_errors)
    
    return mean_result

class ModelNew(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.
    Optimized with custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)