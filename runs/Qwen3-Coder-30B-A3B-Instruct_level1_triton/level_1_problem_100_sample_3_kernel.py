import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute 1 - predictions * targets
    product = predictions * targets
    clamp_input = 1.0 - product
    
    # Apply clamp with min=0
    clamped = tl.maximum(clamp_input, 0.0)
    
    # Store intermediate result
    tl.store(output_ptr + offsets, clamped, mask=mask)

@triton.jit
def mean_reduction_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Sum all elements
    sum_val = tl.sum(input_vals, axis=0)
    
    # Store the sum in the first element
    if block_start == 0:
        tl.atomic_add(output_ptr, sum_val)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    # Allocate intermediate output for clamped values
    clamped_output = torch.empty_like(predictions)
    
    # Grid for clamping operation
    grid_clamp = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch clamping kernel
    hinge_loss_kernel[grid_clamp](
        predictions, targets, clamped_output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # For mean calculation, we'll use a simpler approach with atomic operations
    # since direct reduction in Triton might not be optimal here
    mean_result = torch.sum(clamped_output) / n_elements
    
    return mean_result

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)