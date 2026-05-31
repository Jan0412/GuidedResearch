import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_sq_kernel(
    x_ptr,
    partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_sq = tl.sum(x * x)
    tl.store(partial_sums_ptr + pid, sum_sq)


@triton.jit
def normalize_kernel(
    x_ptr,
    out_ptr,
    norm,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x / norm, mask=mask)


def triton_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    x = x.to(torch.float32)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    partial_sums = torch.zeros(num_blocks, device=x.device, dtype=torch.float32)

    grid = lambda meta: (num_blocks,)
    sum_sq_kernel[grid](x, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    total_sum_sq = torch.sum(partial_sums).item()
    norm = torch.sqrt(torch.tensor(total_sum_sq, device=x.device, dtype=torch.float32))

    out = torch.empty_like(x)
    normalize_kernel[grid](x, out, norm, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_frobenius_norm(x)