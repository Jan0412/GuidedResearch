import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mse_kernel(
    pred_ptr,
    target_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    sq_diff = (pred - target) ** 2
    block_sum = tl.sum(sq_diff)

    tl.atomic_add(out_ptr, block_sum)


def triton_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.is_cuda and target.is_cuda
    pred = pred.contiguous()
    target = target.contiguous()

    n_elements = pred.numel()
    out = torch.zeros((), dtype=torch.float32, device='cuda')

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mse_kernel[grid](pred, target, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out / n_elements


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_mse(predictions, targets)


batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape)*scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []