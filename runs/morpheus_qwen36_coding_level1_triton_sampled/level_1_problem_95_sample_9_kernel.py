import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    logits_ptr, targets_ptr, out_ptr,
    NUM_CLASSES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    logits_row_ptr = logits_ptr + row_idx * NUM_CLASSES
    target_idx = tl.load(targets_ptr + row_idx)

    # Pass 1: Find max for numerical stability
    max_val = -float('inf')
    for offset in range(0, NUM_CLASSES, BLOCK_SIZE):
        indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = indices < NUM_CLASSES
        logits = tl.load(logits_row_ptr + indices, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))

    # Pass 2: Compute sum(exp(x - max))
    sum_exp = 0.0
    for offset in range(0, NUM_CLASSES, BLOCK_SIZE):
        indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = indices < NUM_CLASSES
        logits = tl.load(logits_row_ptr + indices, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(logits - max_val), axis=0)

    logsumexp = tl.log(sum_exp) + max_val
    loss = tl.load(logits_row_ptr + target_idx) - logsumexp
    tl.store(out_ptr + row_idx, loss)

def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert logits.is_cuda and targets.is_cuda
    logits = logits.contiguous()
    targets = targets.contiguous()

    batch_size, num_classes = logits.shape
    out = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)

    BLOCK_SIZE = 128
    grid = (batch_size,)

    cross_entropy_kernel[grid](
        logits, targets, out,
        NUM_CLASSES=num_classes,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.mean()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)