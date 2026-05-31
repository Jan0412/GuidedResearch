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
    
    tl.store(out_ptr + offsets, squared_diff, mask=mask)

@triton.jit
def sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.atomic_add(output_ptr, input_val)

@triton.jit
def mean_kernel(
    sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 1
    
    sum_val = tl.load(sum_ptr + offsets, mask=mask, other=0.0)
    mean_val = sum_val / n_elements
    tl.store(out_ptr + offsets, mean_val, mask=mask)

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # First kernel: compute squared differences
    squared_diff = torch.empty_like(predictions)
    grid1 = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mse_kernel[grid1](predictions, targets, squared_diff, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: compute sum of squared differences
    sum_result = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    grid2 = lambda meta: ((1 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    sum_kernel[grid2](squared_diff, sum_result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Third kernel: compute mean
    mean_result = torch.empty(1, dtype=torch.float32, device=predictions.device)
    grid3 = lambda meta: ((1 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mean_kernel[grid3](sum_result, mean_result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return mean_result

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)