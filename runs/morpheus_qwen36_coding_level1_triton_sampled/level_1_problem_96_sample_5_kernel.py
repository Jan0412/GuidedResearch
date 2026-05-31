import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,
    targets_ptr,
    partial_sums_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    diff = predictions - targets
    abs_diff = tl.abs(diff)
    
    # Smooth L1 Loss: 0.5 * diff^2 if abs_diff < 1 else abs_diff - 0.5
    loss = tl.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    
    local_sum = tl.sum(loss)
    tl.store(partial_sums_ptr + pid, local_sum)


def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    numel = predictions.numel()
    BLOCK_SIZE = 4096
    
    num_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = lambda meta: (num_blocks,)
    
    smooth_l1_loss_kernel[grid](
        predictions_ptr=predictions.data_ptr(),
        targets_ptr=targets.data_ptr(),
        partial_sums_ptr=partial_sums.data_ptr(),
        numel=numel,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return partial_sums.sum() / numel


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)