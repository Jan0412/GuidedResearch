import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the offset for the current program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load the input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply the Tanh activation function
    # Triton's tl.tanh is optimized for GPU hardware
    out = tl.tanh(x)
    
    # Store the result back to memory
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_tanh(x: torch.Tensor):
    """
    Triton wrapper for the tanh operation.
    """
    # Ensure the input is on CUDA and contiguous
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Block size for the kernel
    BLOCK_SIZE = 1024
    
    # Define the grid based on the number of elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Tanh activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return triton_tanh(x)