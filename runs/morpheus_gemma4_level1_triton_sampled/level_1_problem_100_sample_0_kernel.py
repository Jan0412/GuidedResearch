import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load predictions and targets
    p = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Compute hinge loss: max(0, 1 - p * t)
    loss = tl.maximum(0.0, 1.0 - p * t)
    
    # Mask elements outside the boundary to 0 for summation
    loss = tl.where(mask, loss, 0.0)
    
    # Sum the losses within the block
    block_sum = tl.sum(loss, axis=0)
    
    # Atomically add the block sum to the global output
    tl.atomic_add(out_ptr, block_sum)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # Output tensor to store the accumulated sum
    out = torch.zeros((1,), device=predictions.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    hinge_loss_kernel[grid](
        predictions, 
        targets, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute the mean by dividing the total sum by the number of elements
    return out / n_elements

class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.
    Optimized with a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)