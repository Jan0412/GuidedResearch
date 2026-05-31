import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute tanh using Triton's math functions
    out = tl.tanh(x)
    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)

def triton_tanh(x: torch.Tensor):
    """
    Triton-based Tanh implementation
    """
    # Ensure tensor is on GPU and contiguous
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate total number of elements
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    tanh_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Tanh activation
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return triton_tanh(x)