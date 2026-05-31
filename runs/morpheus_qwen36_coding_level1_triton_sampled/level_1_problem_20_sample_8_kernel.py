import torch
import torch.nn as nn
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
    # Calculate the starting offset for this program
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle tensor boundaries
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply LeakyReLU: output = x if x > 0 else negative_slope * x
    out = tl.where(x > 0.0, x, x * negative_slope)

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    """
    Wrapper function to launch the custom Triton LeakyReLU kernel.
    """
    assert x.is_cuda and x.is_contiguous(), "Input tensor must be contiguous and on CUDA."
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Total number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable block size for optimal occupancy
    
    # Define grid configuration
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    leaky_relu_kernel[grid](x, out, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a LeakyReLU activation using a custom Triton kernel.
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
        Applies LeakyReLU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return triton_leaky_relu(x, self.negative_slope)