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
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_classes
    
    # Load predictions for the current row
    predictions = tl.load(predictions_ptr + row_idx * num_classes + offsets, mask=mask, other=float('-inf'))
    
    # Compute max for numerical stability
    max_pred = tl.max(predictions, axis=0)
    
    # Compute log-sum-exp
    shifted_preds = predictions - max_pred
    exp_preds = tl.exp(shifted_preds)
    sum_exp = tl.sum(exp_preds, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_pred
    
    # Get target for the current row
    target = tl.load(targets_ptr + row_idx)
    
    # Compute loss: log_sum_exp - predictions[target]
    loss = log_sum_exp - tl.load(predictions_ptr + row_idx * num_classes + target)
    
    # Store result
    tl.store(out_ptr + row_idx, loss)


def triton_cross_entropy(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size, num_classes = predictions.shape
    out = torch.empty(batch_size, dtype=predictions.dtype, device=predictions.device)
    
    BLOCK_SIZE = 128
    grid = (batch_size,)
    
    cross_entropy_kernel[grid](
        predictions, targets, out,
        batch_size, num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.mean()


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)