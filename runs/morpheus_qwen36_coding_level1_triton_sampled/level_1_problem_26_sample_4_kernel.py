import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,       # Pointer to input tensor
    out_ptr,     # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation using tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Constants for GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute argument for tanh
    tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)
    
    # Compute tanh
    tanh_val = tl.math.tanh(tanh_arg)
    
    # Compute GELU
    out = 0.5 * x * (1.0 + tanh_val)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies GELU activation using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of any shape.
        
    Returns:
        torch.Tensor: Output tensor with GELU applied, same shape as input.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Determine grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a GELU activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)