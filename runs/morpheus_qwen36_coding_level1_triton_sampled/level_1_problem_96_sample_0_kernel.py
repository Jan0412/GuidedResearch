import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr, target_ptr, out_ptr, n_elements, beta, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = pred - target
    abs_diff = tl.abs(diff)
    
    loss = tl.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    
    tl.atomic_add(out_ptr, tl.sum(loss), sem='relaxed')


def triton_smooth_l1_loss(pred, target, beta=1.0):
    assert pred.is_cuda and target.is_cuda
    pred = pred.contiguous()
    target = target.contiguous()
    
    n_elements = pred.numel()
    out = torch.zeros((), device=pred.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    smooth_l1_loss_kernel[grid](pred, target, out, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE)
    
    return out / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)