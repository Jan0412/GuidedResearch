import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * n_cols
    out_row_start_ptr = out_ptr + row_idx * n_cols
    
    # First pass: compute sum of absolute values per row
    acc = 0.0
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_block = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(tl.abs(x_block))
        
    mean = acc / n_cols
    
    # Second pass: normalize and store results
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_block = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_row_start_ptr + offsets, x_block / mean, mask=mask)


def triton_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    l1_normalize_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_normalize(x)