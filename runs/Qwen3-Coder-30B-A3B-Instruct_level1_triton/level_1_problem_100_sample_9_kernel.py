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
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum
    sum_val = tl.sum(x, axis=0)
    
    # Store result
    tl.store(output_ptr, sum_val, mask=mask)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton-based implementation of hinge loss computation.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Prepare output tensor for intermediate computation
    intermediate = torch.empty_like(predictions)
    
    # Number of elements in the tensor
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel for hinge loss computation
    hinge_loss_kernel[grid](predictions, targets, intermediate, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean using another kernel
    mean_result = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # Launch mean kernel
    mean_kernel[1](intermediate, mean_result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Divide by number of elements to get mean
    final_mean = mean_result / n_elements
    
    return final_mean.item()

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for hinge loss computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)