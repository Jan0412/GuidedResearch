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
    BLOCK_SIZE: tl.constexpr
):
    """Compute hinge loss: mean(clamp(1 - predictions * targets, min=0))"""
    # Calculate global offset for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: max(0, 1 - predictions * targets)
    loss = 1.0 - predictions * targets
    loss = tl.maximum(loss, 0.0)
    
    # Store partial sums for reduction
    # For simplicity, we'll store the sum and count in a special format
    # But since we want to compute mean, we can accumulate in a single pass
    tl.store(output_ptr + offsets, loss, mask=mask)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Compute hinge loss using Triton kernel.
    hinge_loss = mean(max(0, 1 - predictions * targets))
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Ensure targets has the same shape as predictions
    if targets.dim() == 1:
        targets = targets.view(-1, 1).expand_as(predictions)
    else:
        targets = targets.contiguous()
    
    # Prepare output tensor for intermediate results
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # For the final mean, we can use PyTorch's efficient implementation
    # after computing the element-wise loss
    temp_output = torch.empty_like(predictions)
    
    # Launch the kernel
    hinge_loss_kernel[grid](
        predictions, targets, temp_output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute mean using PyTorch (efficient reduction)
    return torch.mean(temp_output)


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification tasks
    using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)