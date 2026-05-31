import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Using the numerically stable formulation
    y = 1.0 / (1.0 + tl.exp(-x))
    
    # Store the result
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_sigmoid(x: torch.Tensor):
    """
    Triton implementation of sigmoid activation
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tuned block size for optimal performance
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for sigmoid activation
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