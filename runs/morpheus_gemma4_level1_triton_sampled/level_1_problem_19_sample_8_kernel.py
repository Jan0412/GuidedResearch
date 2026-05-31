import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for the current block
    pid = tl.program_id(0)
    # Compute the range of offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bounds memory access
    mask = offsets < n_elements
    # Load the input values
    x = tl.load(x_ptr + offsets, mask=mask)
    # Apply ReLU: max(0, x)
    out = tl.maximum(0.0, x)
    # Store the result back to memory
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_relu(x: torch.Tensor):
    """
    Custom Triton implementation of the ReLU activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Block size for the kernel (tuned for typical GPU architectures)
    BLOCK_SIZE = 1024
    
    # Grid calculation: number of blocks needed to cover all elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a ReLU activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return triton_relu(x)