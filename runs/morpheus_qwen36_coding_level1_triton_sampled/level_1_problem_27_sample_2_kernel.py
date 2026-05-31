import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def selu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    lambda_ = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    exp_x = tl.exp(x)
    out = tl.where(x > 0.0, x, alpha * (exp_x - 1.0))
    out = lambda_ * out
    
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_selu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_selu(x)