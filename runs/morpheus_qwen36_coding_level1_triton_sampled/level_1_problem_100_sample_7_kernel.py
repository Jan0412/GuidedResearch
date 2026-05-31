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
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    preds = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    val = 1.0 - preds * targets
    val = tl.maximum(val, 0.0)
    block_sum = tl.sum(val, axis=0)
    tl.store(out_ptr, block_sum)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    hinge_loss_kernel[grid](predictions, targets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out.sum() / n_elements


class ModelNew(nn.Module):
    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)