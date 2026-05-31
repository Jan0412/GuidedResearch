import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr, out_ptr, dim, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    base_offset = row_idx * dim

    # Pass 1: Compute sum of absolute values for the row
    sum_abs = 0.0
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_block = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        sum_abs += tl.sum(tl.abs(x_block), axis=0)

    # Pass 2: Normalize (x / mean = x * dim / sum)
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_block = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        out_block = x_block * dim / sum_abs
        tl.store(out_ptr + base_offset + offsets, out_block, mask=mask)


def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size = x.shape[0]
    dim = x.shape[1]
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    l1_norm_kernel[grid](x, out, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)