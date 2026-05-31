import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_kernel(
    pred_ptr, target_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = pred - target
    sq_diff = diff * diff
    
    block_sum = tl.sum(sq_diff)
    tl.store(out_ptr + pid, block_sum)


def triton_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    n_elements = pred.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device=pred.device)
    
    grid = (num_blocks,)
    mse_kernel[grid](pred, target, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return partial_sums.sum() / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)