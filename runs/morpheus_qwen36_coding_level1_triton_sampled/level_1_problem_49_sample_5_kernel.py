import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduce_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * N
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    x_vals = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(x_vals, axis=0)
    tl.store(out_ptr + pid, row_max, mask=pid < M)

def triton_max_reduce(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    x = x.contiguous()
    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 4096
    grid = (M,)
    max_reduce_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(x.shape)
        dims.insert(-1, dims.pop(self.dim))
        x_perm = x.permute(dims).contiguous()
        return triton_max_reduce(x_perm)