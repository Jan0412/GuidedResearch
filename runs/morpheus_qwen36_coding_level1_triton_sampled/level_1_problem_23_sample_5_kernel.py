import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * dim
    out_ptr += row_idx * dim

    # Pass 1: Compute max
    max_val = -float('inf')
    for block_start in range(0, dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x, axis=0))

    # Pass 2: Compute exp and sum
    sum_val = 0.0
    for block_start in range(0, dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        exp_x = tl.exp(x - max_val)
        sum_val += tl.sum(exp_x, axis=0)

    # Pass 3: Normalize
    for block_start in range(0, dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        out = tl.exp(x - max_val) / sum_val
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    softmax_kernel[grid](x, out, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)