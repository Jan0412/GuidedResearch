import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(x_ptr, out_ptr, batch_size, dim, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * dim

    # First pass: compute sum of squares
    sum_sq = 0.0
    start = 0
    while start < dim:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    norm = tl.sqrt(sum_sq)

    # Second pass: normalize and store
    start = 0
    while start < dim:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        out = x / norm
        tl.store(out_ptr + row_start + offsets, out, mask=mask)
        start += BLOCK_SIZE


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    l2_norm_kernel[grid](x, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)