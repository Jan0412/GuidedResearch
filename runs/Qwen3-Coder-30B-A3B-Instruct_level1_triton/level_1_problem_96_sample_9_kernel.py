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
    
    # Load inputs
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute difference
    diff = tl.abs(pred - target)
    
    # Compute smooth L1 loss
    # For |x| < beta: 0.5 * x^2 / beta
    # For |x| >= beta: |x| - 0.5 * beta
    loss = tl.where(
        diff < beta,
        0.5 * diff * diff / beta,
        diff - 0.5 * beta
    )
    
    # Store result
    tl.store(out_ptr + offsets, loss, mask=mask)

@triton.jit
def smooth_l1_loss_backward_kernel(
    pred_ptr,
    target_ptr,
    grad_output_ptr,
    grad_pred_ptr,
    n_elements,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute difference
    diff = pred - target
    
    # Compute gradient
    # For |x| < beta: x / beta
    # For |x| >= beta: sign(x)
    grad = tl.where(
        tl.abs(diff) < beta,
        diff / beta,
        tl.sign(diff)
    )
    
    # Apply gradient from upstream
    grad = grad * grad_out
    
    # Store result
    tl.store(grad_pred_ptr + offsets, grad, mask=mask)

def triton_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0):
    """Compute smooth L1 loss using Triton kernel"""
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    pred = pred.contiguous()
    target = target.contiguous()
    
    # Prepare output tensor for intermediate values
    n_elements = pred.numel()
    BLOCK_SIZE = 1024
    
    # Allocate output tensor for loss values
    loss_vals = torch.empty_like(pred)
    
    # Determine grid
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    smooth_l1_loss_kernel[grid](
        pred, target, loss_vals, n_elements, beta, BLOCK_SIZE
    )
    
    # Return mean of all elements
    return loss_vals.sum() / n_elements

def triton_smooth_l1_loss_backward(pred: torch.Tensor, target: torch.Tensor, grad_output: torch.Tensor, beta: float = 1.0):
    """Compute gradient of smooth L1 loss using Triton kernel"""
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    pred = pred.contiguous()
    target = target.contiguous()
    grad_output = grad_output.contiguous()
    
    # Prepare output tensor for gradients
    n_elements = pred.numel()
    BLOCK_SIZE = 1024
    
    # Allocate output tensor for gradient w.r.t. prediction
    grad_pred = torch.empty_like(pred)
    
    # Determine grid
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch backward kernel
    smooth_l1_loss_backward_kernel[grid](
        pred, target, grad_output, grad_pred, n_elements, beta, BLOCK_SIZE
    )
    
    return grad_pred

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.beta = 1.0

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets, self.beta)