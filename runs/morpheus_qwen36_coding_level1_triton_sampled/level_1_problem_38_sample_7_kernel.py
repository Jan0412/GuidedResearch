import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr, y_ptr,
    row_len,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * row_len

    # First pass: compute sum of absolute values
    row_sum = 0.0
    for col_start in range(0, row_len, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_len
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        row_sum += tl.sum(tl.abs(x))

    # Compute normalization factor (mean of abs values)
    norm_factor = row_sum / row_len

    # Second pass: normalize and store
    for col_start in range(0, row_len, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_len
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        out = x / norm_factor
        tl.store(y_ptr + row_start + offsets, out, mask=mask)

def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    row_len = x.shape[1]
    batch_size = x.shape[0]
    BLOCK_SIZE = 1024

    grid = (batch_size,)
    l1_norm_kernel[grid](x, out, row_len, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)