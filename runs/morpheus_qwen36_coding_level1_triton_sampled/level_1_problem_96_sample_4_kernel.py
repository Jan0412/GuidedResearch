import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr, target_ptr, out_ptr, N, beta, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = pred - target
    abs_diff = tl.abs(diff)
    cond = abs_diff < beta
    
    loss = tl.where(cond, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)
    block_sum = tl.sum(loss)
    tl.store(out_ptr + pid, block_sum)


def triton_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    assert pred.is_cuda and target.is_cuda
    pred = pred.contiguous()
    target = target.contiguous()
    
    N = pred.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = (num_blocks,)
    smooth_l1_loss_kernel[grid](pred, target, partial_sums, N, beta, BLOCK_SIZE=BLOCK_SIZE)
    
    final_loss = partial_sums.sum().item()
    return torch.tensor(final_loss / N, dtype=torch.float32, device='cuda')


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)