import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    sum_sq = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_idx * stride_x + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)

    inv_norm = tl.rsqrt(sum_sq + 1e-5)

    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_idx * stride_x + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + row_idx * stride_x + offsets, x * inv_norm, mask=mask)


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 2048
    grid = (n_rows,)
    l2_norm_kernel[grid](x, out, n_cols, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)