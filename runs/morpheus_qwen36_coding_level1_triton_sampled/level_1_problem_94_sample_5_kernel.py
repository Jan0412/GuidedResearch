import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_kernel(
    predictions_ptr,
    targets_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    diff = preds - targets
    sq_diff = diff * diff
    
    block_sum = tl.sum(sq_diff)
    tl.store(partial_sums_ptr + pid, block_sum)


def triton_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 65536
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = lambda meta: (num_blocks,)
    mse_kernel[grid](
        predictions,
        targets,
        partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return partial_sums.sum() / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)