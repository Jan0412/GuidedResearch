import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
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
    
    # Compute 1 - pred * target
    product = pred * target
    clamp_input = 1.0 - product
    
    # Apply clamp with min=0
    clamped = tl.maximum(clamp_input, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, clamped, mask=mask)

@triton.jit
def mean_reduce_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Use atomic operations to accumulate sum
    tl.atomic_add(output_ptr, tl.sum(input_val, axis=0))

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor for intermediate results
    n_elements = predictions.numel()
    out = torch.empty_like(predictions)
    
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # First compute the hinge loss per element
    hinge_loss_kernel[grid](predictions, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Then compute the mean
    # For this simple case, we'll use a direct approach with reduction
    # Note: For better performance, we could fuse this into one kernel
    total = torch.sum(out)
    mean_result = total / n_elements
    
    return mean_result

class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)