import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for LeakyReLU activation function."""
    # Calculate the block start and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply LeakyReLU: max(x, negative_slope * x) for each element
    # For positive x: output = x
    # For negative x: output = negative_slope * x
    out = tl.where(x > 0, x, x * negative_slope)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    """
    Applies LeakyReLU activation using Triton kernel.
    
    Args:
        x: Input tensor
        negative_slope: The negative slope parameter
        
    Returns:
        Output tensor with LeakyReLU applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Optimal block size for most GPUs
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    leaky_relu_kernel[grid](x, out, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a LeakyReLU activation using Triton kernel.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return triton_leaky_relu(x, self.negative_slope)