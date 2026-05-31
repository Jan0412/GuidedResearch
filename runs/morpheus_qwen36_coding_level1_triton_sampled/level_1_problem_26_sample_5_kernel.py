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
    # Calculate block start
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = 0.70710678118  # 1 / sqrt(2)
    gelu_out = 0.5 * x * (1.0 + tl.math.erf(x * sqrt_2))
    # Store result
    tl.store(out_ptr + offsets, gelu_out, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies GELU activation using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of any shape.
        
    Returns:
        torch.Tensor: Output tensor with GELU applied, same shape as input.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Total number of elements
    n_elements = x.numel()
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
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
        Applies GELU activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)