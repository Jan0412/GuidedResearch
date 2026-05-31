import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_sq_kernel(x_ptr, block_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    local_sum = tl.sum(x * x)
    tl.store(block_sums_ptr + pid, local_sum)


@triton.jit
def normalize_kernel(x_ptr, out_ptr, norm_val, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x / norm_val
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 1024 * 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    block_sums = torch.zeros(num_blocks, dtype=torch.float32, device=x.device)
    
    sum_sq_kernel[(num_blocks,)](x, block_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    
    global_sum = torch.sum(block_sums).item()
    norm_val = global_sum ** 0.5
    
    out = torch.empty_like(x)
    normalize_kernel[(num_blocks,)](x, out, norm_val, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_frobenius_norm(x)