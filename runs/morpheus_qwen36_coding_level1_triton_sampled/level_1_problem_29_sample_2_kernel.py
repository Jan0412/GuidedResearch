import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Numerically stable softplus: log(1 + exp(x))
    # For x > 20, exp(x) overflows FP32, so we safely return x
    out = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softplus(x)