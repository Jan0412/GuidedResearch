import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
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
    
    # Compute softplus with numerical stability
    # For large positive x, softplus(x) ≈ x, so we use x for x > threshold
    # For numerical stability, use:
    # softplus(x) = log(1 + exp(x))
    # When x > 20, exp(x) overflows, so we use x for x > threshold
    # When x < threshold, compute log(1 + exp(x))
    
    threshold = 20.0  # Standard threshold for numerical stability
    is_large = x > threshold
    exp_x = tl.exp(x)
    result = tl.where(is_large, x, tl.log(1.0 + exp_x))
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for Softplus activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)