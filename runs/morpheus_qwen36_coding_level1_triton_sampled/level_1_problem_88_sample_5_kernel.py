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
    # Constants for GELU approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    coeff = 0.044715

    # Program ID gives the starting offset for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute GELU using the tanh approximation
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + coeff * x^3)))
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_val = tl.math.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_val)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton GELU kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size for optimal performance

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super(ModelNew, self).__init__()

    def forward(self, x):
        return triton_gelu(x)