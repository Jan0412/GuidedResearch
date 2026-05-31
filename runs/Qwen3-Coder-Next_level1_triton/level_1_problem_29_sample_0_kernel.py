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
    """Triton kernel for numerical stable softplus activation."""
    # Calculate global offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Numerically stable softplus implementation
    # softplus(x) = log(1 + exp(x))
    # For numerical stability:
    # - If x > 20, exp(x) would overflow, so softplus(x) ≈ x
    # - If x < -20, exp(x) ≈ 0, so softplus(x) ≈ exp(x)
    # - Otherwise use the standard formula with adjustment
    
    # Use the identity: softplus(x) = x + softplus(-x) for x > 0
    # This helps with numerical stability
    threshold = 20.0
    neg_threshold = -20.0
    
    # Compute using stable formulation
    result = tl.where(
        x > threshold,
        x,
        tl.where(
            x < neg_threshold,
            tl.exp(x),
            tl.log1p(tl.exp(x))
        )
    )
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor):
    """
    Triton-based softplus activation with numerical stability.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Common block size for good occupancy
    
    # Determine grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for softplus activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-optimized Softplus activation to the input tensor.
        """
        return triton_softplus(x)