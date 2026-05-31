import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the offset for the current program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to avoid out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load data from the input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Triton's tl.exp handles FP32 operations efficiently
    out = 1.0 / (1.0 + tl.exp(-x))
    
    # Store the result back to the output tensor
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_sigmoid(x: torch.Tensor):
    """
    Triton wrapper for the sigmoid activation function.
    """
    # Ensure input is on CUDA and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Block size can be tuned; 1024 is generally a good starting point for element-wise kernels
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Sigmoid activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return triton_sigmoid(x)