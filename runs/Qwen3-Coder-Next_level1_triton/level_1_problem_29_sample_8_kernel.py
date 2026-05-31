import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    beta: tl.constexpr,  # Softplus beta parameter (default 1.0 in PyTorch)
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
    
    # Compute softplus: log(1 + exp(beta * x))
    # Use the numerically stable formulation:
    # if beta * x > threshold: x
    # else: log(1 + exp(beta * x)) / beta
    # But PyTorch uses beta=1 by default, so we'll implement it directly
    
    # Optimized implementation:
    # softplus(x) = log(1 + exp(x))
    # For numerical stability:
    # if x > threshold: x
    # else: log(1 + exp(x))
    
    # Since we're using beta=1 (default), we'll implement that
    # To match PyTorch behavior, we'll use beta=1.0 and threshold=20.0
    
    # Compute beta * x
    x_scaled = x * beta
    
    # For numerical stability, use the identity:
    # softplus(x) = x if x > threshold
    # softplus(x) = log(1 + exp(x)) if x <= threshold
    result = tl.where(
        x_scaled > threshold,
        x,
        tl.log(1.0 + tl.exp(x_scaled)) / beta
    )
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0):
    """
    This function wraps the Triton kernel call for Softplus activation.
    It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

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
        # Use the default PyTorch parameters: beta=1.0, threshold=20.0
        return triton_softplus(x, beta=1.0, threshold=20.0)