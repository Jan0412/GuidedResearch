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
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: max(0, 1 - predictions * targets)
    loss = 1.0 - predictions * targets
    loss = tl.maximum(loss, 0.0)
    
    # Accumulate for mean calculation - we'll sum here and divide by n_elements later
    tl.store(output_ptr + offsets, loss, mask=mask)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute hinge loss using Triton kernel.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Ensure targets are 1D if needed
    if targets.dim() > 1:
        targets = targets.view(-1)
    
    # Flatten inputs
    n_elements = predictions.numel()
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)
    
    # Prepare output tensor
    out = torch.empty_like(predictions_flat)
    
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    hinge_loss_kernel[grid](predictions_flat, targets_flat, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean
    return torch.mean(out)


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification tasks.
    Uses custom Triton kernel for fused computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)