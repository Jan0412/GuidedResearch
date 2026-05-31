import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output scalar
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss for each element: max(0, 1 - pred * target)
    margin = 1.0 - predictions * targets
    losses = tl.maximum(margin, 0.0)
    
    # Accumulate sum using reduction (we'll do a block-level reduction)
    # Since we're doing a simple sum, we can use tl.sum with a mask
    block_sum = tl.sum(losses, axis=0)
    
    # Store partial sum for this block
    if pid == 0:
        # We'll use a separate kernel for final reduction, so just store partial sums
        tl.store(output_ptr + offsets, losses, mask=mask)
    # For simplicity, we'll use a two-pass approach: first pass computes element-wise loss, second pass sums them


# Optimized version: single kernel that computes mean directly
@triton.jit
def hinge_loss_mean_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    output_ptr,       # Pointer to output scalar (single float)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss for each element: max(0, 1 - pred * target)
    margin = 1.0 - predictions * targets
    losses = tl.maximum(margin, 0.0)
    
    # Accumulate sum using reduction
    block_sum = tl.sum(losses, axis=0)
    
    # Store block sum to temporary buffer (we'll do final reduction in Python or use atomic ops)
    # For simplicity, use atomic add to a global sum (but this can be slow for many blocks)
    # Alternative: use a two-pass approach
    tl.atomic_add(output_ptr, block_sum)


@triton.jit
def finalize_sum_kernel(
    partial_sums_ptr,
    output_ptr,
    n_partial_sums,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums
    
    # Load partial sums
    partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Sum them
    total = tl.sum(partial_sums, axis=0)
    
    # Store final result (only one block should write to output)
    if pid == 0:
        tl.store(output_ptr, total)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute hinge loss: mean(max(0, 1 - predictions * targets))
    Optimized with Triton kernels.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Ensure targets is same shape as predictions (broadcasting handled by PyTorch, but for Triton we need to match)
    # The original code uses broadcasting implicitly, so we'll handle it by flattening both
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)
    
    n_elements = predictions_flat.numel()
    
    # For simplicity, we'll use a two-pass approach:
    # Pass 1: Compute element-wise loss and store in temporary buffer
    # Pass 2: Sum the losses and divide by n_elements
    
    # Temporary buffer for losses
    losses = torch.empty_like(predictions_flat)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Kernel to compute element-wise loss
    @triton.jit
    def compute_losses_kernel(
        predictions_ptr, targets_ptr, losses_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
        
        margin = 1.0 - predictions * targets
        losses = tl.maximum(margin, 0.0)
        
        tl.store(losses_ptr + offsets, losses, mask=mask)
    
    # Launch kernel to compute element-wise losses
    compute_losses_kernel[grid](predictions_flat, targets_flat, losses, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Now compute mean using PyTorch (or we can do it in Triton too)
    # For optimal performance, we can implement a parallel reduction in Triton
    # But for simplicity and correctness, we'll use PyTorch's mean after computing losses
    
    # However, to be fully Triton-based, let's implement a proper reduction:
    # Use a tree-reduction style approach with multiple kernels
    
    # Alternative: use a single kernel with atomic operations (not ideal for large n_elements)
    # Better: use a two-level reduction
    
    # Simple approach: compute sum in Triton, then divide by n_elements
    total_sum = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Use a kernel that accumulates into a single sum using atomics (for simplicity)
    # But for better performance, use a two-pass reduction
    # First pass: compute partial sums per block
    num_blocks = grid({'BLOCK_SIZE': BLOCK_SIZE})[0]
    partial_sums = torch.zeros(num_blocks, device=predictions.device, dtype=predictions.dtype)
    
    @triton.jit
    def sum_losses_kernel(
        losses_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        losses = tl.load(losses_ptr + offsets, mask=mask, other=0.0)
        block_sum = tl.sum(losses, axis=0)
        
        # Store partial sum
        if pid < partial_sums_ptr.numel():
            tl.store(partial_sums_ptr + pid, block_sum)
    
    # Launch the sum kernel
    sum_losses_kernel[(num_blocks,)](losses, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Final reduction: sum the partial sums
    total = torch.sum(partial_sums[:num_blocks])
    
    # Compute mean
    mean_loss = total / n_elements
    return mean_loss


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification tasks using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)