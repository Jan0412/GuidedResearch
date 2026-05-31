import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_last_dim_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, n_cols, BLOCK_SIZE):
        off = start + tl.arange(0, BLOCK_SIZE)
        mask = off < n_cols
        x = tl.load(x_ptr + row_idx * n_cols + off, mask=mask, other=0.0)
        acc += x

    acc = tl.sum(acc)
    acc /= n_cols
    tl.store(out_ptr + row_idx, acc)


def triton_mean_last_dim(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 128
    grid = (n_rows,)
    mean_last_dim_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move the target reduction dimension to the end for contiguous memory access
        x = x.movedim(self.dim, -1)
        return triton_mean_last_dim(x)