import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr, y_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Pass 1: Find max
    max_val = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        block_offsets = start + offsets
        block_mask = block_offsets < n_cols
        x_block = tl.load(row_start_ptr + block_offsets, mask=block_mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_block, axis=0))

    # Pass 2: Compute sum of exp(x - max)
    sum_val = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        block_offsets = start + offsets
        block_mask = block_offsets < n_cols
        x_block = tl.load(row_start_ptr + block_offsets, mask=block_mask, other=0.0)
        exp_x = tl.exp(x_block - max_val)
        sum_val += tl.sum(exp_x, axis=0)

    log_sum_exp = max_val + tl.log(sum_val)

    # Pass 3: Compute log_softmax
    for start in range(0, n_cols, BLOCK_SIZE):
        block_offsets = start + offsets
        block_mask = block_offsets < n_cols
        x_block = tl.load(row_start_ptr + block_offsets, mask=block_mask, other=0.0)
        y_block = x_block - log_sum_exp
        tl.store(y_ptr + row_idx * n_cols + block_offsets, y_block, mask=block_mask)


def triton_log_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous float32 CUDA tensor."
    x = x.contiguous()
    y = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 16384
    grid = (n_rows,)
    log_softmax_kernel[grid](x, y, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x)