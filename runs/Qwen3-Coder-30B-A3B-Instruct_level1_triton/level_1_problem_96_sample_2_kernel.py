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
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = pred - target
    abs_diff = tl.abs(diff)
    
    # Compute smooth L1 loss
    # If |diff| < beta, use squared loss: 0.5 * diff^2 / beta
    # Otherwise, use linear loss: |diff| - 0.5 * beta
    condition = abs_diff < beta
    squared_loss = 0.5 * diff * diff / beta
    linear_loss = abs_diff - 0.5 * beta
    
    loss = tl.where(condition, squared_loss, linear_loss)
    
    # Store the result
    tl.store(out_ptr + offsets, loss, mask=mask)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0):
    """
    Custom Triton implementation of Smooth L1 Loss
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
    smooth_l1_loss_kernel[grid](
        predictions, 
        targets, 
        out, 
        n_elements, 
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return mean of all losses
    return out.mean()

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)