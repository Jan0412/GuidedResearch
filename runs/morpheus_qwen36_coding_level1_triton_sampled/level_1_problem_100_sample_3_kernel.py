import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    preds = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss: max(0, 1 - pred * target)
    loss = tl.maximum(0.0, 1.0 - preds * targets)
    
    # Sum locally across the block
    local_sum = tl.sum(loss)
    
    # Atomically add to the global sum accumulator
    tl.atomic_add(out_ptr, local_sum, sem="relaxed")


def triton_hinge_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert preds.is_cuda and targets.is_cuda
    preds = preds.contiguous()
    targets = targets.contiguous()
    
    n_elements = preds.numel()
    out = torch.zeros(1, dtype=torch.float32, device=preds.device)
    
    BLOCK_SIZE = 512
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_blocks,)
    
    hinge_loss_kernel[grid](preds, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute mean
    return out / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)