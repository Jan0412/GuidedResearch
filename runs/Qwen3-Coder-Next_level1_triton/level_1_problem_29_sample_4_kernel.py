import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute softplus with numerical stability
    # softplus(x) = log(1 + exp(x))
    # For numerical stability:
    # if x > 0: softplus(x) = x + log(1 + exp(-x))
    # if x <= 0: softplus(x) = log(1 + exp(x))
    
    # Use tl.where for conditional computation
    threshold = 20.0  # For x > threshold, exp(x) would overflow, so approximate as x
    
    # More stable implementation
    result = tl.where(
        x > threshold,
        x,  # For large x, softplus(x) ≈ x
        tl.where(
            x > 0,
            x + tl.log(1 + tl.exp(-x)),  # For positive x
            tl.log(1 + tl.exp(x))  # For non-positive x
        )
    )
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor):
    """
    Triton implementation of softplus activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Tunable parameter
    
    # Grid definition
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Softplus activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation using Triton kernel to the input tensor.
        """
        return triton_softplus(x)