import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    # Constants
    CONST_0_5 = 0.5
    CONST_0_7978845608028654 = 0.7978845608028654  # sqrt(2/π)
    CONST_0_044715 = 0.044715
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute inner expression
    inner = CONST_0_7978845608028654 * (x + CONST_0_044715 * x_cubed)
    
    # Compute tanh
    tanh_inner = tl.tanh(inner)
    
    # Final GELU computation
    gelu_out = CONST_0_5 * x * (1.0 + tanh_inner)
    
    # Store result
    tl.store(out_ptr + offsets, gelu_out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    Triton kernel wrapper for GELU activation.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output tensor with GELU applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimized block size for FP32
    
    # Grid calculation
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
        Applies GELU activation using Triton kernel to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)