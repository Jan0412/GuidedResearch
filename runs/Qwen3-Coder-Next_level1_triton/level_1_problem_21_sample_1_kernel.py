import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
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
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Use optimized computation: sigmoid(x) = 1 / (1 + exp(-x))
    # For numerical stability, we can use the identity: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    # But for simplicity and good performance, we'll use the direct formula with clamping
    # To avoid overflow in exp, clamp x to a reasonable range
    # exp(-x) can be very large for negative x, so we use conditional computation
    # However, Triton's math functions are generally well-optimized
    
    # Compute sigmoid using exp(-x) with clamping for stability
    # For large positive x, sigmoid approaches 1, for large negative x, approaches 0
    # We'll use the formula: sigmoid(x) = 1 / (1 + exp(-x))
    # To avoid overflow, we can use: 
    # if x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    # if x < 0: sigmoid(x) = exp(x) / (1 + exp(x))
    
    # Triton has exp, so let's implement stable sigmoid:
    # Using the approach from the reference implementations
    # For numerical stability, split into two cases
    x_clamped = tl.maximum(tl.minimum(x, 20.0), -20.0)  # clamp to avoid overflow
    out = tl.where(x >= 0, 
                   1.0 / (1.0 + tl.exp(-x_clamped)),
                   tl.exp(x_clamped) / (1.0 + tl.exp(x_clamped)))
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for sigmoid activation.
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
    BLOCK_SIZE = 1024  # Tunable parameter for block size, larger for better occupancy

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Sigmoid activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return triton_sigmoid(x)