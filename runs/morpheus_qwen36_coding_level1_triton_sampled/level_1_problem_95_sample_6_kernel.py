import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    predictions_ptr,
    targets_ptr,
    out_ptr,
    num_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = predictions_ptr + pid * num_classes
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    
    x = tl.load(row_ptr + offsets, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    log_sum_exp = tl.log(tl.sum(tl.exp(x - m), axis=0))
    
    target = tl.load(targets_ptr + pid)
    correct_score = tl.load(row_ptr + target)
    
    loss = -correct_score + m + log_sum_exp
    tl.store(out_ptr + pid, loss)

def triton_cross_entropy(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    out = torch.empty(predictions.size(0), dtype=predictions.dtype, device=predictions.device)
    grid = (predictions.size(0),)
    cross_entropy_kernel[grid](predictions, targets, out, num_classes=4096, BLOCK_SIZE=4096)
    return torch.mean(out)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)