import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    
    # Apply swish: x * sigmoid(x)
    out = x * sigmoid_x
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_swish(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Swish activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        return triton_swish(x)