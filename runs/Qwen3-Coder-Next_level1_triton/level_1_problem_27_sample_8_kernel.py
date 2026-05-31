import torch
import torch.nn as nn
import triton
import triton.language as tl

# Constants for SELU
ALPHA = 1.6732632423543772848170429916717
LAMBDA = 1.0507009873554804934193349852946

@triton.jit
def selu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the start index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply SELU: lambda * (alpha * exp(x) - alpha) if x < 0 else lambda * x
    exp_x = tl.exp(x)
    selu_val = LAMBDA * (ALPHA * exp_x - ALPHA)
    # For x >= 0, use LAMBDA * x
    result = tl.where(x < 0, selu_val, LAMBDA * x)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_selu(x: torch.Tensor):
    """
    Apply SELU activation using Triton kernel.
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with SELU applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Reasonable block size for GPU
    
    # Grid: one block per 1024 elements (ceiling division)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for SELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return triton_selu(x)