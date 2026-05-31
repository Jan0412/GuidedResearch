import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * dim
    out_ptr += row_idx * dim

    sum_abs = 0.0
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        sum_abs += tl.sum(abs_x)

    mean_abs = sum_abs / dim
    inv_mean = 1.0 / mean_abs

    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        out = x * inv_mean
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    dim = x.shape[1]
    batch_size = x.shape[0]
    BLOCK_SIZE = 2048

    grid = (batch_size,)
    l1_norm_kernel[grid](x, out, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)