import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID and calculate the range of indices this block handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Softplus formula: log(1 + exp(x))
    # For numerical stability, we use the identity: log(1 + exp(x)) approx x for large x
    # Typically, for x > 20, exp(x) is very large and log(1 + exp(x)) is practically x.
    out = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_softplus(x: torch.Tensor):
    """
    Triton wrapper for the softplus activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure the tensor is contiguous for the kernel
    x = x.contiguous()
    n_elements = x.numel()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Block size for the Triton kernel
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)