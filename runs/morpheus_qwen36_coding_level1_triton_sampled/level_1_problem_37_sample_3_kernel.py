import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def norm_kernel(
    x_ptr,
    sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x
    local_sum = tl.sum(x_sq)
    tl.atomic_add(sum_ptr, local_sum)


@triton.jit
def scale_kernel(
    x_ptr,
    out_ptr,
    inv_norm,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * inv_norm
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        n_elements = x.numel()
        BLOCK_SIZE = 128

        sum_ptr = torch.zeros(1, dtype=torch.float32, device=x.device)

        grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
        norm_kernel[grid](x, sum_ptr, n_elements, BLOCK_SIZE=BLOCK_SIZE)

        sum_sq = sum_ptr.item()
        inv_norm = 1.0 / torch.sqrt(torch.tensor(sum_sq, device=x.device)).item()

        out = torch.empty_like(x)
        scale_kernel[grid](x, out, inv_norm, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out