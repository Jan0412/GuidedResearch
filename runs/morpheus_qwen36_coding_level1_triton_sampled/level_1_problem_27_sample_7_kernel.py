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
    SCALE = 1.0507009873554804934193349852946
    ALPHA = 1.6732632423543772848170429916717

    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    exp_x = tl.exp(x)
    out = tl.where(x > 0, SCALE * x, ALPHA * SCALE * (exp_x - 1.0))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_selu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32, "Input must be a contiguous FP32 CUDA tensor."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 2048

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_selu(x)


batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []