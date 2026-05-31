import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program's starting offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute HardSigmoid: max(0, min(1, (x + 3) / 6))
    # HardSigmoid formula: clamp((x + 3) / 6, 0, 1)
    temp = (x + 3.0) / 6.0
    out = tl.where(temp < 0.0, 0.0, tl.where(temp > 1.0, 1.0, temp))
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-based HardSigmoid implementation.
    
    Args:
        x: Input tensor
        
    Returns:
        Output tensor with HardSigmoid applied
    """
    # Ensure input is contiguous and on CUDA
    assert x.is_cuda, "Tensor must be on CUDA device."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Reasonable default block size for GPU
    
    # Calculate grid size
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs HardSigmoid activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation using Triton kernel to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return triton_hardsigmoid(x)