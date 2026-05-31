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
    
    # Compute squared difference
    diff = pred - target
    squared_diff = diff * diff
    
    # Store intermediate result
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
    
    # Load input
    val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store for reduction
    tl.store(output_ptr + offsets, val, mask=mask)

@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean_val = val / n_elements
    
    # Store result
    tl.store(output_ptr + offsets, mean_val, mask=mask)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Computes MSE using Triton kernels
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Get total number of elements
    n_elements = predictions.numel()
    
    # Allocate output tensor for squared differences
    squared_diff = torch.empty_like(predictions)
    
    # Compute squared differences
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    mse_kernel[grid](predictions, targets, squared_diff, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all elements
    sum_result = torch.sum(squared_diff)
    
    # Compute mean
    mean_result = sum_result / n_elements
    
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