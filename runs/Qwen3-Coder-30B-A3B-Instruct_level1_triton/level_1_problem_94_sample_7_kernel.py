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
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce sum
    sum_val = tl.sum(x, axis=0)
    
    # Store result
    tl.store(out_ptr, sum_val, mask=mask)

@triton.jit
def mean_kernel(
    sum_ptr,
    count,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Load sum
    sum_val = tl.load(sum_ptr)
    
    # Compute mean
    mean_val = sum_val / count
    
    # Store result
    tl.store(out_ptr, mean_val)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Custom Triton implementation of MSE loss computation
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor for squared differences
    squared_diff = torch.empty_like(predictions)
    
    # Get total number of elements
    n_elements = predictions.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Grid size calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Step 1: Compute squared differences
    mse_kernel[grid](predictions, targets, squared_diff, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 2: Compute sum of squared differences
    sum_tensor = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    sum_kernel[1](squared_diff, sum_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Step 3: Compute mean
    mean_tensor = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    mean_kernel[1](sum_tensor, n_elements, mean_tensor, BLOCK_SIZE=BLOCK_SIZE)
    
    return mean_tensor

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for MSE loss computation
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)