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
def mean_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Use atomic operation to accumulate sum
    tl.atomic_add(output_ptr, tl.sum(input_val, axis=0))

def triton_mse(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # First kernel: compute squared differences
    squared_diff = torch.empty_like(predictions)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    mse_kernel[grid](predictions, targets, squared_diff, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: compute mean
    mean_result = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    mean_kernel[grid](squared_diff, mean_result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Normalize by number of elements
    return mean_result / n_elements

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)