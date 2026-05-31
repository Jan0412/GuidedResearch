import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = x * x * x
    sqrt_2_pi = tl.math.sqrt(2.0 / tl.math.pi)
    arg = sqrt_2_pi * (x + 0.044715 * x3)
    tanh_val = tl.math.tanh(arg)
    out = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        return triton_gelu(x)

def get_inputs():
    return [torch.rand(8192, 8192)]

def get_init_inputs():
    return []