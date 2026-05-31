import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    sum_sq = 0.0
    num_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    i = 0
    while i < num_blocks:
        col_offsets = i * BLOCK_SIZE + offsets
        row_mask = col_offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=row_mask, other=0.0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
        i += 1

    norm = tl.sqrt(sum_sq)
    norm = tl.where(norm == 0.0, 1.0, norm)

    i = 0
    while i < num_blocks:
        col_offsets = i * BLOCK_SIZE + offsets
        row_mask = col_offsets < n_cols
        x_vals = tl.load(x_ptr + row_start + col_offsets, mask=row_mask, other=0.0)
        out_vals = x_vals / norm
        tl.store(out_ptr + row_start + col_offsets, out_vals, mask=row_mask)
        i += 1


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    l2_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)