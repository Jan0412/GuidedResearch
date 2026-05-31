import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr, out_ptr, dim_size, stride, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride
    acc = 1.0
    for start in range(0, dim_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=1.0)
        x = x * acc
        out = tl.cumprod(x, axis=0)
        tl.store(out_ptr + row_start + offsets, out, mask=mask)
        
        block_prod = 1.0
        for i in range(0, BLOCK_SIZE):
            block_prod *= x[i]
        acc *= block_prod


def triton_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    dim_size = x.shape[dim]
    stride = x.stride(dim)
    num_slices = x.numel() // dim_size
    BLOCK_SIZE = 1024
    grid = (num_slices,)
    cumprod_kernel[grid](x, out, dim_size, stride, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)