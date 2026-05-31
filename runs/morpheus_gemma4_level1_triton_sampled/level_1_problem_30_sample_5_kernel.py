import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softsign_kernel(
    x_ptr,        # Pointer to input tensor
    out_ptr,      # Pointer to output tensor
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # This kernel performs element-wise softsign: x / (1 + abs(x))
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute softsign
    # Formula: x / (1 + |x|)
    res = x / (1.0 + tl.abs(x))
    
    # Store the result
    tl.store(out_ptr + offsets, res, mask=mask)

def triton_softsign(x: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper for the Softsign activation function.
    """
    # Ensure input is contiguous on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Block size for the kernel; 1024 is typically a good default for element-wise ops
    BLOCK_SIZE = 1024
    
    # Grid is 1D across the total number of elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    softsign_kernel[grid](
        x, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softsign activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softsign activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape on CUDA.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        return triton_softsign(x)