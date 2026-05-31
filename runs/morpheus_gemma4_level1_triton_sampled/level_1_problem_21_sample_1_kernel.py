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
    # Program ID
    pid = tl.program_id(0)
    # Compute offsets for the current block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to prevent out-of-bounds access
    mask = offsets < n_elements
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute sigmoid: 1 / (1 + exp(-x))
    out = 1.0 / (1.0 + tl.exp(-x))
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_sigmoid(x: torch.Tensor):
    """
    Triton implementation of the sigmoid activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    
    # Define grid
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