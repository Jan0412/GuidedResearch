import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_kernel(
    pred_ptr,
    target_ptr,
    partial_sums_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    diff = pred - target
    sq_diff = diff * diff
    partial_sum = tl.sum(sq_diff)

    tl.store(partial_sums_ptr + pid, partial_sum)


def triton_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert predictions.is_cuda and targets.is_cuda, "Inputs must be on CUDA"
    predictions = predictions.contiguous()
    targets = targets.contiguous()

    num_elements = predictions.numel()
    if num_elements == 0:
        return torch.tensor(0.0, device=predictions.device)

    BLOCK_SIZE = 256
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device=predictions.device)

    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mse_kernel[grid](predictions, targets, partial_sums, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    return partial_sums.sum() / num_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)