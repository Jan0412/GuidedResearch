import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
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
    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    # For numerical stability, use the identity: sigmoid(x) = 1 / (1 + exp(-x))
    # For large positive x, exp(-x) underflows to 0, so sigmoid(x) ≈ 1
    # For large negative x, exp(-x) overflows, so we use sigmoid(x) = exp(x) / (1 + exp(x))
    # Triton provides tl.sigmoid for this purpose, but we can implement it efficiently:
    # Using the stable implementation: sigmoid(x) = 0.5 * (1 + tanh(x * 0.5))
    # Or use tl.where for better numerical stability
    exp_neg_x = tl.exp(-x)
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    # Compute swish: x * sigmoid(x)
    swish_out = x * sigmoid_x
    # Store the result
    tl.store(out_ptr + offsets, swish_out, mask=mask)


def triton_swish(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for Swish activation.
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
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Swish activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor using custom Triton kernel.
        """
        return triton_swish(x)