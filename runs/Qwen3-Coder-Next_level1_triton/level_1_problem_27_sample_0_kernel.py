import torch
import torch.nn as nn
import triton
import triton.language as tl

# Constants for SELU
SELU_ALPHA = 1.6732632423543772848170429916717
SELU_LAMBDA = 1.0507009873554804934193349852946

@triton.jit
def selu_kernel(
    x_ptr,          # Pointer to input
    out_ptr,        # Pointer to output
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SELU: λ * (α * exp(x) - α) if x < 0, else λ * x
    # Using the formula: selu(x) = λ * (α * exp(x) - α) for x < 0, λ * x for x >= 0
    exp_x = tl.exp(x)
    selu_val = tl.where(
        x < 0,
        SELPU_LAMBDA * (SELU_ALPHA * exp_x - SELU_ALPHA),
        SELU_LAMBDA * x
    )
    
    # Store result
    tl.store(out_ptr + offsets, selu_val, mask=mask)


def triton_selu(x: torch.Tensor):
    """
    Triton implementation of SELU activation.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output tensor with SELU applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Tunable parameter
    
    # Grid definition
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
        Applies SELU activation using custom Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return triton_selu(x)