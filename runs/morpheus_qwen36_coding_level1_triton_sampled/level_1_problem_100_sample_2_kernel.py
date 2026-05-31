import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    out_ptr,
    N,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    row_indices = offsets // D
    targets = tl.load(targets_ptr + row_indices, mask=mask, other=0.0)

    loss = tl.maximum(0.0, 1.0 - predictions * targets)
    block_sum = tl.sum(loss)

    tl.store(out_ptr + tl.program_id(0), block_sum)

def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    N = predictions.numel()
    D = predictions.shape[-1]
    BLOCK_SIZE = 256

    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_sums = torch.empty(num_blocks, dtype=torch.float32, device='cuda')

    grid = lambda meta: (num_blocks,)
    hinge_loss_kernel[grid](predictions, targets, block_sums, N, D, BLOCK_SIZE)

    total_sum = block_sums.sum().item()
    return total_sum / N

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)