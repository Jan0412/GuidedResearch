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
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute Tanh
    # tl.tanh is available in triton.language
    out = tl.tanh(x)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_tanh(x: torch.Tensor):
    """
    Wrapper for the Triton tanh kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Block size can be tuned; 1024 is typically a good starting point for element-wise kernels
    BLOCK_SIZE = 1024
    
    # Grid is calculated to cover all elements in the tensor
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
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
        Applies Tanh activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return triton_tanh(x)