import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    predictions_ptr, targets_ptr, out_ptr,
    batch_size, num_classes,
    stride_pred, stride_target,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    pred_row_ptr = predictions_ptr + pid * stride_pred
    target = tl.load(targets_ptr + pid * stride_target)
    
    # Pass 1: Find max for numerical stability
    max_val = -float('inf')
    for i in range(0, num_classes, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        preds = tl.load(pred_row_ptr + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(preds, axis=0))
        
    # Pass 2: Compute sum(exp(x - max))
    sum_exp = 0.0
    for i in range(0, num_classes, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        preds = tl.load(pred_row_ptr + offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(preds - max_val), axis=0)
        
    log_sum_exp = tl.log(sum_exp) + max_val
    loss = log_sum_exp - tl.load(pred_row_ptr + target)
    
    tl.store(out_ptr + pid, loss)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[1]
    
    out = torch.empty(batch_size, dtype=torch.float32, device=predictions.device)
    
    BLOCK_SIZE = 128
    grid = lambda meta: (batch_size,)
    
    cross_entropy_kernel[grid](
        predictions, targets, out,
        batch_size, num_classes,
        predictions.stride(0), targets.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        row_losses = triton_cross_entropy(predictions, targets)
        return row_losses.sum() / predictions.shape[0]


def get_inputs():
    batch_size = 32768
    num_classes = 4096
    return [torch.rand(batch_size, num_classes), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []