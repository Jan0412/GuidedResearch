import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation: 0.5 * x * (1.0 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    # Precomputed constant: sqrt(2/π) ≈ 0.7978845608028654
    c = 0.7978845608028654
    
    # Compute x³
    x_cubed = x * x * x
    # Compute inner term: x + 0.044715 * x³
    inner = x + 0.044715 * x_cubed
    # Compute tanh argument: sqrt(2/π) * (x + 0.044715 * x³)
    tanh_arg = c * inner
    # Compute tanh
    tanh_val = tl.tanh(tanh_arg)
    # Compute final GELU: 0.5 * x * (1.0 + tanh_val)
    out = 0.5 * x * (1.0 + tanh_val)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for GELU activation.
    It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        # Instead of using PyTorch's implementation, call our Triton-based GELU
        return triton_gelu(x)