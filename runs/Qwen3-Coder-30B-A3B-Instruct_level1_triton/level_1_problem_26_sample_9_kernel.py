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
    # Compute the block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation using tanh
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute the argument of tanh
    arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    
    # Apply GELU formula
    tanh_arg = tl.tanh(arg)
    gelu_result = 0.5 * x * (1.0 + tanh_arg)
    
    # Store the result
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

def triton_gelu(x: torch.Tensor):
    """
    Triton implementation of GELU activation function.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Calculate total number of elements
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model with custom Triton GELU kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation using Triton kernel for speedup.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)