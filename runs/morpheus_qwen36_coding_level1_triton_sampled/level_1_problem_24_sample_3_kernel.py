import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    max_val = -float('inf')
    sum_val = 0.0

    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        new_max = tl.max(x, axis=0)
        sum_val = tl.exp(max_val - new_max) * sum_val + tl.sum(tl.exp(x - new_max))
        max_val = tl.maximum(max_val, new_max)

    log_sum = tl.log(sum_val)

    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        out = x - max_val - log_sum
        tl.store(out_ptr + row_start + offsets, out, mask=mask)

def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert dim == 1, "Currently optimized for dim=1."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x, self.dim)