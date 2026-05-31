import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * dim
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    max_val = tl.max(x, axis=0)
    y = x - max_val
    sum_val = tl.sum(tl.exp(y), axis=0)
    out = y - tl.log(sum_val)
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


def triton_log_softmax(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = dim
    grid = (batch_size,)
    log_softmax_kernel[grid](x, out, dim, BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x)