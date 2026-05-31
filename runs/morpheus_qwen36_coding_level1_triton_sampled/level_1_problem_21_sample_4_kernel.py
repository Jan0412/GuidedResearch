import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Numerically stable sigmoid optimized for FP32
    sig = tl.where(
        x >= 0,
        1.0 / (1.0 + tl.exp(-x)),
        tl.exp(x) / (1.0 + tl.exp(x))
    )
    tl.store(out_ptr + offsets, sig, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)