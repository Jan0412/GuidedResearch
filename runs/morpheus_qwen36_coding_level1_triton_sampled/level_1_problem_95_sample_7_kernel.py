import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_loss_kernel(
    predictions_ptr,
    targets_ptr,
    losses_ptr,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    predictions_ptr += batch_idx * num_classes
    targets_ptr += batch_idx
    losses_ptr += batch_idx
    
    target = tl.load(targets_ptr).to(tl.int32)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    preds = tl.load(predictions_ptr + offsets, mask=mask, other=-float('inf'))
    
    max_val = tl.max(preds, axis=0)
    exp_preds = tl.exp(preds - max_val)
    sum_exp = tl.sum(exp_preds, axis=0)
    log_sum_exp = tl.log(sum_exp)
    
    target_pred = tl.load(predictions_ptr + target)
    loss = -(target_pred - max_val) + log_sum_exp
    
    tl.store(losses_ptr, loss)

def triton_cross_entropy(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[1]
    
    losses = torch.empty(batch_size, dtype=predictions.dtype, device=predictions.device)
    
    BLOCK_SIZE = 4096
    
    grid = (batch_size,)
    cross_entropy_loss_kernel[grid](
        predictions, targets, losses, num_classes, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return losses.mean()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)