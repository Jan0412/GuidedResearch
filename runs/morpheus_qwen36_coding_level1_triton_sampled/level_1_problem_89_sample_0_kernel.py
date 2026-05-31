import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for start in tl.static_range(0, n_cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < n_cols
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        acc = acc + x
        tl.store(y_ptr + row_start + idx, acc, mask=mask)


def triton_cumsum(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    assert dim == 1, "Only dim=1 is supported in this optimized kernel."
    
    x = x.contiguous().float()
    y = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 256
    
    grid = (n_rows,)
    cumsum_kernel[grid](x, y, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum(x, self.dim)