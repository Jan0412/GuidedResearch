import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute difference
    diff = pred - target
    abs_diff = tl.abs(diff)
    
    # Compute smooth L1 loss
    # If |diff| < beta, use quadratic term: 0.5 * diff^2 / beta
    # Otherwise, use linear term: |diff| - 0.5 * beta
    quadratic = 0.5 * diff * diff / beta
    linear = abs_diff - 0.5 * beta
    
    # Select appropriate computation based on condition
    loss = tl.where(abs_diff < beta, quadratic, linear)
    
    # Store result
    tl.store(out_ptr + offsets, loss, mask=mask)

@triton.jit
def reduce_sum_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce sum within block
    reduced = tl.sum(input_vals, axis=0)
    
    # Store partial sum
    tl.store(output_ptr + tl.program_id(0), reduced, mask=tl.program_id(0) < (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    """
    Compute smooth L1 loss using Triton kernels.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape."
    
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
    smooth_l1_loss_kernel[grid](predictions, targets, out, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all elements to compute final loss
    total_elements = n_elements
    while total_elements > 1:
        temp_out = torch.zeros((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, dtype=torch.float32, device=predictions.device)
        reduce_grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        reduce_sum_kernel[reduce_grid](out, temp_out, total_elements, BLOCK_SIZE=BLOCK_SIZE)
        out = temp_out
        total_elements = out.numel()
    
    # Return mean loss
    return out[0] / n_elements

class ModelNew(nn.Module):
    """
    An optimized version of the Smooth L1 Loss model using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)