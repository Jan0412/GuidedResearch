import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr,  # Pointer to predictions
    target_ptr,  # Pointer to targets
    out_ptr,  # Pointer to output (scalar loss)
    n_elements,  # Total number of elements
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Accumulator for the total loss
    acc_sum = tl.zeros((1,), dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load inputs
        pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        
        # Compute difference
        diff = pred - target
        abs_diff = tl.abs(diff)
        
        # Smooth L1 loss: 
        # if |x| < beta: 0.5 * x^2 / beta
        # else: |x| - 0.5 * beta
        loss = tl.where(
            abs_diff < beta,
            0.5 * diff * diff / beta,
            abs_diff - 0.5 * beta
        )
        
        # Accumulate sum using masked load to avoid out-of-bounds
        acc_sum += tl.sum(tl.where(mask, loss, 0.0))
    
    # Store result (averaged over n_elements)
    mean_loss = acc_sum / n_elements
    tl.store(out_ptr, mean_loss)


def triton_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0):
    """
    Computes Smooth L1 (Huber) loss using Triton kernel.
    """
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    assert pred.shape == target.shape, "Predictions and targets must have the same shape."
    
    pred = pred.contiguous()
    target = target.contiguous()
    
    n_elements = pred.numel()
    BLOCK_SIZE = 1024
    
    # Output tensor (scalar)
    out = torch.empty(1, device=pred.device, dtype=pred.dtype)
    
    # Launch kernel
    grid = (1,)
    smooth_l1_loss_kernel[grid](pred, target, out, n_elements, beta=beta, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks using optimized Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        # Use the Triton-based Smooth L1 loss implementation
        return triton_smooth_l1_loss(predictions, targets, beta=1.0)