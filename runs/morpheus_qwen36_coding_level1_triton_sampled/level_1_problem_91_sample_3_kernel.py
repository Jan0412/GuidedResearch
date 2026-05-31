import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr, out_ptr, stride_x, stride_out, N, D, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    x_ptr += pid * stride_x
    out_ptr += pid * stride_out

    acc = 0.0
    num_blocks = (D + BLOCK_SIZE - 1) // BLOCK_SIZE

    for k in range(num_blocks):
        idx = num_blocks - 1 - k
        start = idx * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D

        x_block = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        block_rev_cumsum = tl.flip(tl.cumsum(tl.flip(x_block, axis=0), axis=0), axis=0)
        out_block = block_rev_cumsum + acc
        tl.store(out_ptr + offsets, out_block, mask=mask)
        acc += tl.sum(x_block, axis=0)


def triton_reverse_cumsum(x: torch.Tensor, dim: int = 1):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    N, D = x.shape
    BLOCK_SIZE = 1024
    grid = (N,)
    reverse_cumsum_kernel[grid](x, out, x.stride(0), out.stride(0), N, D, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim == 1:
            return triton_reverse_cumsum(x, dim=self.dim)
        return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)