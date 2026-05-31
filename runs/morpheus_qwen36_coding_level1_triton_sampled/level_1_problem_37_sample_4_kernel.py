import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def reduce_sum_sq_kernel(x_ptr, sum_sq_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sum_sq_block = tl.sum(x * x)
    tl.atomic_add(sum_sq_ptr, sum_sq_block, sem="relaxed")

@triton.jit
def div_kernel(x_ptr, out_ptr, sum_sq_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    norm = tl.sqrt(tl.load(sum_sq_ptr))
    out = x / norm
    tl.store(out_ptr + offsets, out, mask=mask)

def frobenius_norm_normalize(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    num_elements = x.numel()
    sum_sq = torch.zeros(1, dtype=torch.float32, device='cuda')
    BLOCK_SIZE = 2048
    
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    reduce_sum_sq_kernel[grid](x, sum_sq, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    out = torch.empty_like(x)
    div_kernel[grid](x, out, sum_sq, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frobenius_norm_normalize(x)

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return []