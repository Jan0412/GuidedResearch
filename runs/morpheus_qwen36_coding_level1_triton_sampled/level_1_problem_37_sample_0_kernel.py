import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(
    x_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x
    sum_sq = tl.sum(x_sq)
    tl.store(partial_sums_ptr + pid, sum_sq)


@triton.jit
def reduce_partial_sums_kernel(
    partial_sums_ptr, total_sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    total = 0.0
    start = 0
    while start < n_elements:
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
        total += tl.sum(vals)
        start += BLOCK_SIZE
    if pid == 0:
        tl.store(total_sum_ptr, total)


@triton.jit
def normalize_kernel(
    x_ptr, out_ptr, inv_norm_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    inv_norm = tl.load(inv_norm_ptr)
    out = x * inv_norm
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be contiguous FP32 CUDA tensor."
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    # Pass 1: Compute partial sums across blocks
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device=x.device)
    grid1 = lambda meta: (num_blocks,)
    partial_sum_kernel[grid1](x, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Pass 2: Reduce partial sums to a single scalar value
    total_sum = torch.empty(1, dtype=torch.float32, device=x.device)
    grid2 = lambda meta: (1,)
    reduce_partial_sums_kernel[grid2](partial_sums, total_sum, num_blocks, BLOCK_SIZE=BLOCK_SIZE)

    # Compute inverse norm to avoid division
    inv_norm = 1.0 / torch.sqrt(total_sum)

    # Pass 3: Apply normalization
    out = torch.empty_like(x)
    grid3 = lambda meta: (num_blocks,)
    normalize_kernel[grid3](x, out, inv_norm, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_frobenius_norm(x)