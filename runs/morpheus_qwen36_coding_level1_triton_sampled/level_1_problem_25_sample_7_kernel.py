import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute Swish: x * sigmoid(x)
    # Triton's tl.sigmoid is optimized for FP32
    sig = tl.sigmoid(x)
    out = x * sig
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the Triton Swish kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Total number of elements
    n_elements = x.numel()
    
    # Tunable block size; power of 2 is recommended
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for Swish activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        return triton_swish(x)