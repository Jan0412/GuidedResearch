import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softsign_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines the block of data this instance handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Softsign calculation: x / (1 + abs(x))
    out = x / (1.0 + tl.abs(x))

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_softsign(x: torch.Tensor):
    """
    Triton wrapper for the softsign activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure input is contiguous for efficient pointer arithmetic
    x = x.contiguous()
    n_elements = x.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    
    # Grid is 1D since it's an element-wise operation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    softsign_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
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
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        return triton_softsign(x)