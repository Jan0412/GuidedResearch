import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardsigmoid_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute HardSigmoid: max(0, min(1, (x + 1) / 2))
    # First compute (x + 1) / 2
    x_scaled = (x + 1.0) * 0.5
    
    # Apply clamp: max(0, min(1, x_scaled))
    clipped = tl.minimum(tl.maximum(x_scaled, 0.0), 1.0)
    
    # Store result
    tl.store(output_ptr + offsets, clipped, mask=mask)

def triton_hardsigmoid(x: torch.Tensor):
    """
    Triton implementation of HardSigmoid activation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Block size tuned for performance
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    hardsigmoid_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for HardSigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return triton_hardsigmoid(x)