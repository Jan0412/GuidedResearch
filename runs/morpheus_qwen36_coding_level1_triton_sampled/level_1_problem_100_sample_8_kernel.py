import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_kernel(
    pred_ptr, target_ptr, out_ptr,
    batch_size, input_dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * input_dim

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + (offsets // input_dim), mask=mask, other=0.0)

    val = 1.0 - pred * target
    relu_val = tl.maximum(val, 0.0)

    block_sum = tl.sum(relu_val)
    tl.atomic_add(out_ptr, block_sum, sem="relaxed")


def triton_hinge_loss(predictions, targets):
    assert predictions.is_cuda and targets.is_cuda
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    batch_size, input_dim = predictions.shape
    total_elements = batch_size * input_dim
    out = torch.zeros((), device=predictions.device, dtype=torch.float32)

    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    hinge_loss_kernel[grid](predictions, targets, out, batch_size, input_dim, BLOCK_SIZE=BLOCK_SIZE)
    return out / total_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)