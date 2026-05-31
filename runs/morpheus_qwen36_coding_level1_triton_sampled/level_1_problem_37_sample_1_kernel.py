import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_sq_kernel(x_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_sq = tl.sum(x * x)
    tl.store(partial_sums_ptr + pid, sum_sq)


@triton.jit
def reduce_kernel(partial_sums_ptr, out_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    vals = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(vals)
    if pid == 0:
        tl.store(out_ptr, total)


@triton.jit
def normalize_kernel(x_ptr, out_ptr, norm_inv, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * norm_inv
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fro_norm(x: torch.Tensor):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Pass 1: Compute sum of squares per block
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.empty(num_blocks, dtype=torch.float32, device=x.device)
    sum_sq_kernel[(num_blocks,)](x, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Pass 2: Reduce partial sums to a global scalar
    reduce_num_blocks = (num_blocks + BLOCK_SIZE - 1) // BLOCK_SIZE
    global_sum_sq = torch.empty(1, dtype=torch.float32, device=x.device)
    reduce_kernel[(reduce_num_blocks,)](partial_sums, global_sum_sq, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute inverse norm factor
    norm_inv = 1.0 / torch.sqrt(global_sum_sq).item()
    
    # Pass 3: Elementwise division by the norm
    out = torch.empty_like(x)
    normalize_kernel[(num_blocks,)](x, out, norm_inv, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_fro_norm(x)