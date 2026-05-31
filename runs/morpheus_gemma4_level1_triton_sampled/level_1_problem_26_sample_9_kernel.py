import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
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
    # Mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU exact formula: 0.5 * x * (1 + erf(x / sqrt(2)))
    # 1/sqrt(2) is approximately 0.7071067811865476
    res = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    
    # Store result
    tl.store(out_ptr + offsets, res, mask=mask)

def triton_gelu(x: torch.Tensor):
    """
    Triton wrapper for the GELU activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure input is contiguous for the Triton kernel
    x = x.contiguous()
    n_elements = x.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    
    # Define grid based on total elements and block size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a GELU activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)