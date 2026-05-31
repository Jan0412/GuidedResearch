import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def elu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ELU: x if x > 0 else alpha * (exp(x) - 1)
    out = tl.where(x > 0, x, alpha * (tl.exp(x) - 1))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_elu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 2048
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    elu_kernel[grid](x, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_elu(x, self.alpha)