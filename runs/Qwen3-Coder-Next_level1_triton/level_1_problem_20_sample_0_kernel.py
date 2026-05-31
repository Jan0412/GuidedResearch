import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    negative_slope,  # Negative slope parameter
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply LeakyReLU: max(x, negative_slope * x)
    out = tl.where(x > 0, x, x * negative_slope)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01):
    """
    Applies LeakyReLU activation using Triton kernel.
    
    Args:
        x: Input tensor
        negative_slope: The negative slope parameter for LeakyReLU
    
    Returns:
        Output tensor with LeakyReLU applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimal block size for FP32 operations
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    leaky_relu_kernel[grid](x, out, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs LeakyReLU activation using custom Triton kernel.
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
        return triton_leaky_relu(x, negative_slope=self.negative_slope)