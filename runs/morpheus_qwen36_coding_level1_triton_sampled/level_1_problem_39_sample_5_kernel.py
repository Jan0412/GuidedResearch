import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * dim
    x_ptr += row_offset
    out_ptr += row_offset

    # First pass: compute sum of squares per row
    sum_sq = 0.0
    start = 0
    while start < dim:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
        start += BLOCK_SIZE

    # Compute L2 norm and handle zero norm case
    norm = tl.sqrt(sum_sq)
    norm = tl.where(norm == 0.0, 1.0, norm)

    # Second pass: divide by norm and store result
    start = 0
    while start < dim:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x / norm, mask=mask)
        start += BLOCK_SIZE


def triton_l2_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be contiguous FP32 CUDA tensor"
    x = x.contiguous()
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    l2_norm_kernel[grid](x, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x)