import torch
import torch.nn as nn
import triton
import triton.language as tl


# Constants for SELU
ALPHA = 1.6732632423543772
LAMBDA = 1.0507009873554805


@triton.jit
def selu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
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
    
    # SELU implementation: λ * (α * exp(x) - α) if x <= 0, λ * x if x > 0
    # Using branchless implementation for better performance:
    # selu(x) = λ * (max(x, 0) + α * exp(min(x, 0)) - α)
    # But for clarity and performance, we'll use conditional logic
    
    # Compute exponential part for negative values
    exp_x = tl.exp(x)
    
    # SELU logic: if x <= 0: λ * (α * exp(x) - α), else: λ * x
    out = tl.where(
        x <= 0,
        LAMBDA * (ALPHA * exp_x - ALPHA),
        LAMBDA * x
    )
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_selu(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for SELU activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return triton_selu(x)