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
    max_val = float('-inf')
    for start in range(0, N, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
    tl.store(out_ptr + pid, max_val)

def triton_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    x_moved = x.movedim(dim, -1).contiguous()
    M = x_moved.shape[:-1].numel()
    N = x_moved.shape[-1]
    out = torch.empty(x_moved.shape[:-1], device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = (M,)
    max_reduce_kernel[grid](x_moved, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_reduce(x, self.dim)