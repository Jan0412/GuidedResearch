import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_partial_kernel(
    pred_ptr, target_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    diff = pred - target
    sq = diff * diff
    
    block_sum = tl.sum(sq)
    
    tl.store(partial_sums_ptr + pid, block_sum)


def triton_mse(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 1024
    
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    mse_partial_kernel[grid](
        predictions, targets, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    total_sum = partial_sums.sum()
    return total_sum / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)