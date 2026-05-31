import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    pred_ptr, target_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    val = 1.0 - pred * target
    val = tl.maximum(val, 0.0)
    val = tl.where(mask, val, 0.0)
    
    sum_val = tl.sum(val)
    tl.store(out_ptr + pid, sum_val)


def triton_hinge_loss(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Handle broadcasting by flattening both tensors to 1D
    pred_flat = predictions.flatten()
    tgt_flat = targets.expand(predictions.shape).flatten()
    n_elements = pred_flat.numel()
    
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device='cuda')
    
    grid = (num_blocks,)
    
    hinge_loss_kernel[grid](
        pred_flat, tgt_flat, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    total_sum = partial_sums.sum().item()
    return total_sum / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)