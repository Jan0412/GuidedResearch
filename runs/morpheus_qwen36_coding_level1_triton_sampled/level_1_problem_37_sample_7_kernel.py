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
def normalize_kernel(x_ptr, out_ptr, inv_sqrt_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    inv_sqrt = tl.load(inv_sqrt_ptr)
    out = x * inv_sqrt
    tl.store(out_ptr + offsets, out, mask=mask)

def frobenius_norm_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device='cuda')

    grid_sum = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    sum_sq_kernel[grid_sum](x, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    total_sum_sq = torch.sum(partial_sums)
    inv_sqrt = 1.0 / torch.sqrt(total_sum_sq)

    grid_norm = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    normalize_kernel[grid_norm](x, out, inv_sqrt, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frobenius_norm_triton(x)

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return []