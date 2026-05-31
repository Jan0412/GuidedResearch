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
    # Compute program ID
    pid = tl.program_id(0)
    # Compute offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute softplus with numerical stability
    # softplus(x) = log(1 + exp(x))
    # For numerical stability:
    # - If x > 20, exp(x) overflows, so softplus(x) ≈ x
    # - If x < -20, exp(x) underflows to 0, so softplus(x) ≈ exp(x)
    threshold = 20.0
    exp_x = tl.exp(x)
    result = tl.where(
        x > threshold,
        x,
        tl.where(
            x < -threshold,
            tl.exp(x),
            tl.log(1.0 + exp_x)
        )
    )
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor):
    """
    Applies Softplus activation using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output tensor with Softplus applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Tunable parameter
    
    # Determine grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Softplus activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)