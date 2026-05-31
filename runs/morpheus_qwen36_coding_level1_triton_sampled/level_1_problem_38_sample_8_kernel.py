import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    dim,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * dim
    out_row_ptr = out_ptr + row_idx * dim
    
    sum_abs = 0.0
    
    # First pass: compute sum of absolute values
    start = 0
    while start < dim:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        sum_abs += tl.sum(tl.abs(x), axis=0)
        start += BLOCK_SIZE
        
    # Compute scale factor: x / mean_abs = x * dim / sum_abs
    scale = dim / sum_abs
    
    # Second pass: scale the row
    start = 0
    while start < dim:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        out = x * scale
        tl.store(out_row_ptr + offsets, out, mask=mask)
        start += BLOCK_SIZE


def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, dim = x.shape
    BLOCK_SIZE = 1024
    
    grid = (batch_size,)
    l1_norm_kernel[grid](x, out, dim, batch_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)