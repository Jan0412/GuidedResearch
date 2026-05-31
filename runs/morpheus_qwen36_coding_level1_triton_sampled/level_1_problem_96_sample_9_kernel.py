import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
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
    
    block_sum = tl.sum(loss, axis=0)
    
    tl.store(out_ptr + pid, block_sum)


def triton_smooth_l1_loss(predictions, targets, beta=1.0):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    partial_sums = torch.empty(grid_size, dtype=torch.float32, device='cuda')
    
    smooth_l1_loss_kernel[grid_size](
        predictions, targets, partial_sums, n_elements, beta, BLOCK_SIZE=BLOCK_SIZE
    )
    
    total_loss = partial_sums.sum()
    mean_loss = total_loss / n_elements
    
    return mean_loss


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)