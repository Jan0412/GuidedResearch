import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
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
    # Compute x³
    x_cubed = x * x * x
    # Compute x + 0.044715 * x³
    inner = x + 0.044715 * x_cubed
    # Compute sqrt(2/π) * inner
    scaled_inner = inner * tl.math.sqrt(2.0 / math.pi)
    # Compute tanh of scaled inner
    tanh_val = tl.math.tanh(scaled_inner)
    # Compute final GELU: 0.5 * x * (1.0 + tanh_val)
    out = 0.5 * x * (1.0 + tanh_val)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for GELU activation.
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
    """
    Optimized implementation of GELU using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x):
        return triton_gelu(x)