import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    sum_abs = 0.0
    for start_col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        sum_abs += tl.sum(tl.abs(x_vals))

    mean_abs = sum_abs / n_cols

    for start_col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + col_offsets, x_vals / mean_abs, mask=mask)


def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_rows, n_cols = x.shape
    BLOCK_SIZE = 128

    grid = lambda meta: (n_rows,)

    l1_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)