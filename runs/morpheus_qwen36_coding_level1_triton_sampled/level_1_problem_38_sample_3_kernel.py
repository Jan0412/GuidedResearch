import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_normalize_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # First pass: compute sum of absolute values
    sum_abs = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_cols - start)
        x = tl.load(x_ptr + row_idx * n_cols + start + offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        sum_abs += tl.sum(abs_x)
        
    mean_abs = sum_abs / n_cols
    mean_abs = tl.where(mean_abs == 0, 1.0, mean_abs)
    
    # Second pass: normalize
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < (n_cols - start)
        x = tl.load(x_ptr + row_idx * n_cols + start + offsets, mask=mask, other=0.0)
        out = x / mean_abs
        tl.store(out_ptr + row_idx * n_cols + start + offsets, out, mask=mask)


def triton_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size = x.shape[0]
    n_cols = x.shape[1]
    BLOCK_SIZE = 1024
    
    grid = (batch_size,)
    l1_normalize_kernel[grid](x, out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_normalize(x)