import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offset vector
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid elements
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply HardSigmoid: max(0, min(1, (x + 3) / 6))
    out = tl.where(x < -3.0, 0.0, tl.where(x > 3.0, 1.0, (x + 3.0) / 6.0))
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton HardSigmoid kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable block size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for HardSigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)