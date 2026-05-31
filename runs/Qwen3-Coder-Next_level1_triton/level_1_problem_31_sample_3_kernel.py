import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def elu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    alpha_ptr,  # Pointer to alpha (scalar)
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
    # ELU: x if x > 0 else alpha * (exp(x) - 1)
    # Use tl.where for vectorized conditional
    alpha = tl.load(alpha_ptr)
    out = tl.where(x > 0, x, alpha * (tl.exp(tl.where(x > -100, x, -100)) - 1))
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_elu(x: torch.Tensor, alpha: float = 1.0):
    """
    This function wraps the Triton kernel call for ELU activation. It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Create alpha tensor on GPU
    alpha_tensor = torch.tensor([alpha], device=x.device, dtype=x.dtype)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    elu_kernel[grid](x, out, alpha_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs an ELU activation using Triton kernel.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the optimized ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        return triton_elu(x, alpha=self.alpha)