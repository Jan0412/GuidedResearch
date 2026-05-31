import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        x = tl.load(x_ptr + row_start * N + start + offsets, mask=mask, other=0.0)
        acc += x
    mean_val = tl.sum(acc) / N
    tl.store(out_ptr + pid, mean_val)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    # Move reduction dimension to the last axis for efficient block processing
    x_moved = x.movedim(dim, -1).contiguous()
    M = x_moved.numel() // x_moved.size(-1)
    N = x_moved.size(-1)
    
    out = torch.empty(x_moved.shape[:-1], dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (M,)
    
    mean_kernel[grid](x_moved, out, M, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Move the dimension back to its original position
    out_moved = out.movedim(-1, dim)
    return out_moved


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_mean(x, self.dim)