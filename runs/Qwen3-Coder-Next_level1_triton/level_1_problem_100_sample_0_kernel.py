import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    predictions_ptr,  # Pointer to predictions tensor
    targets_ptr,      # Pointer to targets tensor
    out_ptr,          # Pointer to output (loss value)
    n_elements,       # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load predictions and targets
    preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: max(0, 1 - pred * target)
    # We use tl.maximum for the clamp operation
    loss = tl.maximum(0.0, 1.0 - preds * targets)
    
    # Store the loss values
    tl.store(out_ptr + offsets, loss, mask=mask)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """
    Triton-based Hinge Loss computation.
    
    This function computes the hinge loss in a single kernel, avoiding 
    intermediate tensor allocations and leveraging Triton's efficiency
    for element-wise operations.
    """
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Ensure targets is shaped like predictions (batch_size,)
    if targets.dim() == 1 and targets.shape[0] == predictions.shape[0]:
        targets = targets.view(-1, 1) if predictions.dim() > 1 else targets
    
    # Prepare output tensor for intermediate loss values
    loss_out = torch.empty_like(predictions)
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    hinge_loss_kernel[grid](predictions, targets, loss_out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean using torch (this is a reduction operation that Triton can handle, 
    # but for simplicity and numerical stability, we use torch.mean here)
    return torch.mean(loss_out)


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification tasks using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)