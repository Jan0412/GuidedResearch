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
    
    # Compute absolute difference
    diff = tl.abs(pred - target)
    
    # Compute smooth L1 loss
    # If |diff| < beta: 0.5 * (diff / beta)^2 * beta
    # Else: |diff| - 0.5 * beta
    loss = tl.where(
        diff < beta,
        0.5 * (diff / beta) * (diff / beta) * beta,
        diff - 0.5 * beta
    )
    
    # Store the result
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
    
    # Load input values
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute difference
    diff = pred - target
    
    # Compute gradient w.r.t. prediction
    # If |diff| < beta: sign(diff) * diff / beta
    # Else: sign(diff)
    grad = tl.where(
        tl.abs(diff) < beta,
        diff / beta,
        tl.sign(diff)
    )
    
    # Apply gradient from upstream
    grad = grad * grad_out
    
    # Store the result
    tl.store(grad_pred_ptr + offsets, grad, mask=mask)

def triton_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0):
    """Compute smooth L1 loss using Triton kernel"""
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    pred = pred.contiguous()
    target = target.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(pred)
    
    # Number of elements in the tensor
    n_elements = pred.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    smooth_l1_loss_kernel[grid](
        pred, target, out, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.sum()

def triton_smooth_l1_loss_backward(pred: torch.Tensor, target: torch.Tensor, grad_output: torch.Tensor, beta: float = 1.0):
    """Compute backward pass of smooth L1 loss using Triton kernel"""
    assert pred.is_cuda and target.is_cuda, "Tensors must be on CUDA."
    pred = pred.contiguous()
    target = target.contiguous()
    grad_output = grad_output.contiguous()
    
    # Prepare output tensor
    grad_pred = torch.empty_like(pred)
    
    # Number of elements in the tensor
    n_elements = pred.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    smooth_l1_loss_backward_kernel[grid](
        pred, target, grad_output, grad_pred, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE
    )
    return grad_pred

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    Optimized with custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)