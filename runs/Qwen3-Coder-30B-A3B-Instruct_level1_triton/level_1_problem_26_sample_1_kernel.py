import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using constants for better accuracy
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute the argument for tanh
    arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    
    # Apply GELU approximation
    tanh_arg = tl.tanh(arg)
    gelu_result = 0.5 * x * (1.0 + tanh_arg)
    
    # Store result
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

def triton_gelu(x: torch.Tensor):
    """
    Triton implementation of GELU activation function.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tuned block size for good performance
    
    # Grid configuration
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for GELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)