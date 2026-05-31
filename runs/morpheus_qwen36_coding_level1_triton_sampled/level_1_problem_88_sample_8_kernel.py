import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    c1 = 0.7978845608028654  # sqrt(2.0 / math.pi)
    c2 = 0.044715
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused GELU approximation: 0.5 * x * (1.0 + tanh(c1 * (x + c2 * x^3)))
    x3 = x * x * x
    inner = x + c2 * x3
    tanh_val = tl.math.tanh(c1 * inner)
    out = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return triton_gelu(x)