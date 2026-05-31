import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply LeakyReLU: max(negative_slope * x, x)
    scaled_x = x * negative_slope
    output = tl.maximum(scaled_x, x)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01):
    """
    Triton implementation of LeakyReLU activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tuned block size for better performance
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    leaky_relu_kernel[grid](
        x, 
        out, 
        n_elements, 
        negative_slope, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for LeakyReLU activation.
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