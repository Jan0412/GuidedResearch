import torch
import torch.nn as nn
import triton
import triton.language as tl

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
    
    # Compute absolute difference
    diff = tl.abs(pred - target)
    
    # Compute smooth L1 loss
    # For |diff| < beta: 0.5 * (diff / beta)^2
    # For |diff| >= beta: |diff| - 0.5 * beta
    condition = diff < beta
    squared_term = 0.5 * (diff / beta) * (diff / beta)
    linear_term = diff - 0.5 * beta
    loss = tl.where(condition, squared_term, linear_term)
    
    # Store the result
    tl.store(out_ptr + offsets, loss, mask=mask)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    """
    Triton implementation of Smooth L1 Loss for regression tasks.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape."
    
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
    
    # Return mean of all losses
    return tl.sum(out, axis=0) / n_elements

class ModelNew(nn.Module):
    """
    An optimized model that computes Smooth L1 (Huber) Loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)