import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    row_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * row_size
    sum_sq = 0.0
    for start in range(0, row_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_size
        x = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
    inv_norm = tl.rsqrt(sum_sq)
    for start in range(0, row_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_size
        x = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        out = x * inv_norm
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, row_size = x.shape
    BLOCK_SIZE = 128
    grid = (batch_size,)
    l2_norm_kernel[grid](x, out, row_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)