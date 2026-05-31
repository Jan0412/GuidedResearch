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
    # Calculate the offset for the current program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)

    # GELU Tanh Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Constants for GELU approximation
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715

    # Compute the cubic term and the inner part of tanh
    x_cube = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cube)
    
    # Compute GELU
    out = 0.5 * x * (1.0 + tl.tanh(inner))

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_gelu(x: torch.Tensor):
    """
    Triton wrapper for the GELU activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    n_elements = x.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Tuning parameter for block size
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
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
        Applies GELU activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)