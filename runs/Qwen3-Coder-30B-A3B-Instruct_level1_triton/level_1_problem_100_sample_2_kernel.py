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
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute 1 - predictions * targets
    intermediate = 1.0 - predictions * targets
    
    # Apply clamp(min=0) - equivalent to max(0, intermediate)
    clamped = tl.maximum(intermediate, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, clamped, mask=mask)

@triton.jit
def mean_reduction_kernel(
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
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Sum reduction
    sum_val = tl.sum(input_vals, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + tl.program_id(0), sum_val, mask=tl.program_id(0) < tl.cdiv(n_elements, BLOCK_SIZE))

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton implementation of hinge loss computation
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(predictions)
    
    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    hinge_loss_kernel[grid](predictions, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean using another kernel
    # First compute sum
    partial_sums = torch.zeros((triton.cdiv(n_elements, BLOCK_SIZE),), device=predictions.device, dtype=torch.float32)
    mean_reduction_kernel[triton.cdiv(n_elements, BLOCK_SIZE)](out, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reduce partial sums to final mean
    total_sum = partial_sums.sum()
    mean_result = total_sum / n_elements
    
    return mean_result

class ModelNew(nn.Module):
    """
    Optimized Model using Triton kernels for hinge loss computation
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)