import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(x_ptr, out_ptr, row_stride, dim, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_sum = tl.zeros((), dtype=tl.float32)
    
    # First pass: compute sum of absolute values for the row
    for start in range(0, dim, BLOCK_SIZE):
        offsets = row_idx * row_stride + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        row_sum += tl.sum(tl.abs(x), axis=0)
    
    mean = row_sum / dim
    
    # Second pass: normalize the row
    for start in range(0, dim, BLOCK_SIZE):
        offsets = row_idx * row_stride + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = x / mean
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 1024
    
    grid = (batch_size,)
    l1_norm_kernel[grid](x, out, x.stride(0), dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_l1_norm(x)