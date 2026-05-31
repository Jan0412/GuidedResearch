import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(
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
    # Compute tanh using the formula: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    # For numerical stability, we use the identity: tanh(x) = sign(x) * (1 - 2/(e^(2*|x|) + 1))
    # This avoids overflow for large |x|
    
    # For better numerical stability, use the approximation for large |x|
    # tanh(x) ≈ 1 - 2*exp(-2x) for x > 0, and tanh(x) ≈ -1 + 2*exp(2x) for x < 0
    # But Triton has built-in tanh function which is optimized
    
    # Compute tanh directly using Triton's math functions
    out = tl.math.tanh(x)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_tanh(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for tanh activation.
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
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Tanh activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return triton_tanh(x)