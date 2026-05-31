import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def swish_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Swish activation: x * sigmoid(x)
    # tl.sigmoid is available in triton.language
    out = x * tl.sigmoid(x)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_swish(x: torch.Tensor):
    """
    Triton wrapper for the Swish activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure tensor is contiguous for efficient memory access in Triton
    x = x.contiguous()
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Block size for element-wise operation
    BLOCK_SIZE = 1024
    
    # Grid calculation: one block per BLOCK_SIZE elements
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    swish_kernel[grid](
        x, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Swish activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        return triton_swish(x)