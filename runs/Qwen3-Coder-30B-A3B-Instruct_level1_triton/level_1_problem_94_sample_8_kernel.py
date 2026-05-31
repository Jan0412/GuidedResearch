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
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = pred - target
    squared_diff = diff * diff
    
    # Use atomic operation to accumulate sum
    tl.atomic_add(out_ptr, tl.sum(squared_diff, axis=None), sem="acq_rel")

@triton.jit
def mean_kernel(
    sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 1  # Only one element for the final mean
    
    # Load the sum from global memory
    sum_val = tl.load(sum_ptr, mask=mask, other=0.0)
    mean_val = sum_val / n_elements
    
    # Store the final mean
    tl.store(out_ptr, mean_val, mask=mask)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # Allocate intermediate tensor for sum
    sum_tensor = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    BLOCK_SIZE = 1024
    
    # Grid for sum computation
    grid_sum = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch sum kernel
    mse_kernel[grid_sum](
        predictions,
        targets,
        sum_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute final mean
    mean_tensor = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    grid_mean = lambda meta: ((1 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    mean_kernel[grid_mean](
        sum_tensor,
        mean_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return mean_tensor.item()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)