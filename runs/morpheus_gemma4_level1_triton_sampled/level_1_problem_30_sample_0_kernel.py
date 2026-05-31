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
    # Program ID
    pid = tl.program_id(0)
    # Offset for the current block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for out-of-bounds access
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute Softsign: x / (1 + |x|)
    out = x / (1.0 + tl.abs(x))
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_softsign(x: torch.Tensor):
    """
    Triton wrapper for the Softsign activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # BLOCK_SIZE is a tunable parameter; 1024 is generally a good starting point for element-wise ops
    BLOCK_SIZE = 1024
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
        Applies Softsign activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        return triton_softsign(x)