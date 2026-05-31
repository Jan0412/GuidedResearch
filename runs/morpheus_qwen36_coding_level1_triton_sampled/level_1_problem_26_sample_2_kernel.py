import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation constants
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    
    x_cubed = x * x * x
    arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_val = tl.math.tanh(arg)
    
    out = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gelu(x)