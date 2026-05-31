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
    """
    Triton kernel for GELU activation using the tanh approximation:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute GELU approximation
    # Constants for tanh approximation
    # sqrt(2/pi) approx 0.7978845608
    x_cube = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x_cube)
    out = 0.5 * x * (1.0 + tl.tanh(inner))
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_gelu(x: torch.Tensor):
    """
    Wrapper for the Triton GELU kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # Tuning block size for memory throughput
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    gelu_kernel[grid](
        x, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a GELU activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)