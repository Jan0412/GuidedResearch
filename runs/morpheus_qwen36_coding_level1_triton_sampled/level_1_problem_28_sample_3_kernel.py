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
    # Calculate the starting offset for this program
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle elements beyond tensor size
    mask = offsets < n_elements
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute HardSigmoid: clamp((x + 3) / 6, 0, 1)
    # Step 1: Shift and scale
    val = (x + 3.0) / 6.0
    # Step 2: Clamp to [0, 1] using min/max for compatibility
    val = tl.minimum(val, 1.0)
    val = tl.maximum(val, 0.0)
    # Store result
    tl.store(out_ptr + offsets, val, mask=mask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton HardSigmoid kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Tunable block size
    BLOCK_SIZE = 128
    
    # Grid configuration
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for HardSigmoid.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)