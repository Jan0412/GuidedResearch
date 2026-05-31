import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def frobenius_norm_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Accumulate the sum of squares
    total_sum = 0.0
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        total_sum += tl.sum(x * x)
    # Compute the reciprocal of the square root of the total sum
    inv_norm = 1.0 / tl.sqrt(total_sum)
    # Scale the input and store
    for start in range(0, n_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = x * inv_norm
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter
    grid = (1,)  # Single program for simplicity
    frobenius_norm_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_frobenius_norm(x)