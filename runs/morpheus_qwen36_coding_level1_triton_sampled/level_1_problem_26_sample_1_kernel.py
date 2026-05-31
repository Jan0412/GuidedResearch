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
    # Calculate block start and offset range
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask for boundary elements
    mask = offsets < n_elements
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    tanh_arg = sqrt_2_over_pi * (x + coeff * x * x * x)
    out = 0.5 * x * (1.0 + tl.math.tanh(tanh_arg))
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Wraps the Triton GELU kernel for PyTorch integration.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for element-wise FP32 operations
    
    # Grid calculation: 1D grid of blocks
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gelu(x)