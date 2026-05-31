import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    predictions_ptr,
    targets_ptr,
    out_ptr,
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    target = tl.load(targets_ptr + pid)
    
    max_val = -float('inf')
    sum_exp = 0.0
    
    for offset in range(0, num_classes, BLOCK_SIZE):
        indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = indices < num_classes
        vals = tl.load(predictions_ptr + pid * num_classes + indices, mask=mask, other=-float('inf'))
        
        chunk_max = tl.max(vals, axis=0)
        new_max = tl.maximum(max_val, chunk_max)
        
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(vals - new_max), axis=0)
        max_val = new_max
        
    loss = max_val + tl.log(sum_exp) - tl.load(predictions_ptr + pid * num_classes + target)
    tl.store(out_ptr + pid, loss)


def triton_cross_entropy(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, num_classes = predictions.shape
    out = torch.empty(batch_size, dtype=predictions.dtype, device=predictions.device)
    
    grid = (batch_size,)
    cross_entropy_kernel[grid](
        predictions, targets, out,
        batch_size, num_classes,
        BLOCK_SIZE=128, num_warps=4, num_stages=2
    )
    return out.sum() / batch_size


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)