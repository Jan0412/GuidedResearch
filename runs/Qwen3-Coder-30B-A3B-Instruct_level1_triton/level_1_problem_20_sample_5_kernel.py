import torch
import torch.nn as nn
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
    # Compute block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply LeakyReLU: max(x, negative_slope * x)
    output = tl.where(x > 0, x, x * negative_slope)
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    Triton implementation of LeakyReLU activation.
    """
    # Ensure tensor is on GPU and contiguous
    x = x.contiguous().to(torch.float32)
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate total number of elements
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    leaky_relu_kernel[grid](x, output, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

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
        Applies LeakyReLU activation using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return triton_leaky_relu(x, self.negative_slope)