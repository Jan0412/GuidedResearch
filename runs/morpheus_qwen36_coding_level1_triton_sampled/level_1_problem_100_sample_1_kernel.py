import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    pred_ptr, target_ptr, block_sums_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    preds = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute hinge loss element-wise: max(0, 1 - pred * target)
    loss = tl.maximum(0.0, 1.0 - preds * targets)
    
    # Reduce sum over the block
    block_sum = tl.sum(loss)
    tl.store(block_sums_ptr + pid, block_sum)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    BLOCK_SIZE = 128
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    block_sums = torch.zeros(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = lambda meta: (num_blocks,)
    hinge_loss_kernel[grid](predictions, targets, block_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return block_sums.sum() / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)