import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    beta: tl.constexpr,  # Beta parameter for softplus (default 1.0)
    threshold: tl.constexpr,  # Threshold for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute scaled input: beta * x
    scaled_x = beta * x
    
    # Use numerically stable implementation of softplus:
    # softplus(x) = log(1 + exp(x))
    # For large positive x, use x (since log(1 + exp(x)) ≈ x)
    # For large negative x, use log(1 + exp(x)) directly but avoid overflow
    result = tl.where(
        scaled_x > threshold,
        scaled_x / beta,  # For large positive values, approximates to x
        tl.log1p(tl.exp(scaled_x)) / beta  # For other values, use numerically stable formula
    )
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor, beta: float = 1.0):
    """
    This function wraps the Triton kernel call for Softplus activation.
    
    Args:
        x (torch.Tensor): Input tensor
        beta (float): Beta parameter for softplus (default 1.0)
        
    Returns:
        torch.Tensor: Output tensor with Softplus applied, same shape as input.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Threshold for numerical stability (commonly used threshold is 20)
    threshold = 20.0

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softplus_kernel[grid](x, out, n_elements, beta=beta, threshold=threshold, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using Triton kernel.
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