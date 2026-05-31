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
    # This kernel handles a block of elements in the input tensor
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply LeakyReLU: x if x >= 0 else negative_slope * x
    out = tl.where(x >= 0, x, x * negative_slope)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_leaky_relu(x: torch.Tensor, negative_slope: float):
    """
    Triton wrapper for the LeakyReLU operation.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Use a block size of 1024 for FP32 efficiency
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
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
        Applies LeakyReLU activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return triton_leaky_relu(x, self.negative_slope)