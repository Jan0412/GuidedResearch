import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(x_ptr, out_ptr, dim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * dim
    sum_abs = 0.0
    
    num_blocks = (dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_chunk = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        sum_abs += tl.sum(tl.abs(x_chunk))
        
    mean = sum_abs / dim
    
    for block_idx in range(num_blocks):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_chunk = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        out = x_chunk / mean
        tl.store(out_ptr + row_start + offsets, out, mask=mask)


def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    l1_norm_kernel[grid](x, out, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)