import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(x_ptr, out_ptr, row_size, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * row_size
    out_ptr += row_idx * row_size

    num_blocks = (row_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    sum_sq = 0.0
    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    inv_sqrt = tl.rsqrt(sum_sq + 1e-8)

    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = x * inv_sqrt
        tl.store(out_ptr + offsets, out, mask=mask)

def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    l2_norm_kernel[grid](x, out, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []