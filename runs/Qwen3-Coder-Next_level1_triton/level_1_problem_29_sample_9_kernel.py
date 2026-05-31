import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for softplus activation: log(1 + exp(x))"""
    # Compute global thread index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute softplus with numerical stability
    # Use the identity: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    # Or for better stability: if x > 20, softplus(x) ≈ x; if x < -20, softplus(x) ≈ exp(x)
    threshold = 20.0
    x_abs = tl.abs(x)
    x_sign = tl.where(x >= 0, 1.0, -1.0)
    
    # For numerical stability, use piecewise approximation
    # When x > threshold: softplus(x) ≈ x
    # When x < -threshold: softplus(x) ≈ exp(x)
    # Otherwise: use standard formula
    result = tl.where(
        x > threshold,
        x,
        tl.where(
            x < -threshold,
            tl.exp(x),
            tl.where(
                x >= 0,
                x + tl.log(1 + tl.exp(-x)),
                tl.log(1 + tl.exp(x))
            )
        )
    )
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softplus activation using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output tensor with softplus applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    
    # Block size for Triton kernel
    BLOCK_SIZE = 512
    
    # Grid: one block per 512 elements (rounded up)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernel for softplus activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softplus activation to the input tensor using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)