import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(
    x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    # Calculate offsets for the current block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute HardSigmoid: (x + 3) / 6 clamped to [0, 1]
    y = (x + 3.0) / 6.0
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    
    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the custom HardSigmoid Triton kernel.
    """
    assert x.is_cuda and x.is_contiguous(), "Input tensor must be contiguous and on CUDA."
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tuned for high throughput on modern GPUs
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for HardSigmoid activation.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)