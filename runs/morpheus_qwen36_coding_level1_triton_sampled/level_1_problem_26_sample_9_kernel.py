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
    # Create offsets for the current block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to handle elements beyond the tensor size
    mask = offsets < n_elements
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    tanh_input = sqrt_2_over_pi * (x + coeff * x * x * x)
    tanh_out = tl.math.tanh(tanh_input)
    out = 0.5 * x * (1.0 + tanh_out)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies GELU activation using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Output tensor with GELU applied.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size
    
    # Grid calculation: one block per grid dimension
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernel for GELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with GELU applied.
        """
        return triton_gelu(x)