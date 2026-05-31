import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_kernel(
    x_ptr, out_ptr, M, stride_x, stride_out, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * stride_x
    out_row_ptr = out_ptr + row_idx * stride_out
    
    acc = 1.0
    start = 0
    while start < M:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < M
        x = tl.load(row_ptr + offsets, mask=mask, other=1.0)
        acc = acc * x
        tl.store(out_row_ptr + offsets, acc, mask=mask)
        start += BLOCK_SIZE


def triton_cumprod(x: torch.Tensor, dim: int):
    assert x.is_cuda and dim == 1, "Tensors must be on CUDA and dim must be 1 for this kernel."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    N = x.shape[0]
    M = x.shape[1]
    BLOCK_SIZE = 1024
    
    grid = (N,)
    cumprod_kernel[grid](x, out, M, x.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)