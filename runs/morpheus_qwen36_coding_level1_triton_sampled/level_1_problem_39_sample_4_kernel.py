import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    sum_sq = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, 1e-12)
    
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        out = x / norm
        tl.store(out_ptr + row_start + offsets, out, mask=mask)


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_rows,)
    l2_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)