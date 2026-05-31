import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(pred_ptr, target_ptr, out_ptr, n_elements, D, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    preds = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    # Handle broadcasting: target index repeats every D elements
    target_offsets = offsets // D
    targets = tl.load(target_ptr + target_offsets, mask=mask, other=0.0)
    
    # Compute hinge loss term: max(0, 1 - pred * target)
    vals = 1.0 - preds * targets
    vals = tl.maximum(vals, 0.0)
    
    # Sum over the block
    block_sum = tl.sum(vals)
    
    # Atomic add to global sum
    tl.atomic_add(out_ptr, 0, block_sum)


def triton_hinge_loss(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    D = predictions.shape[-1]
    
    out = torch.zeros(1, device=predictions.device, dtype=torch.float32)
    
    BLOCK_SIZE = 4096
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    hinge_loss_kernel[grid](predictions, targets, out, n_elements, D, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean
    return out / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)