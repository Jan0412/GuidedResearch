import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply HardTanh: clamp between min_val and max_val
    out = tl.where(x < min_val, min_val, tl.where(x > max_val, max_val, x))
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    """
    This function wraps the Triton kernel call for HardTanh activation.
    
    Args:
        x (torch.Tensor): Input tensor on CUDA.
        min_val (float): Minimum value for clamping.
        max_val (float): Maximum value for clamping.
        
    Returns:
        torch.Tensor: Output tensor with HardTanh applied, same shape as input.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    hardtanh_kernel[grid](x, out, n_elements, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a HardTanh activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardTanh applied, same shape as input.
        """
        return triton_hardtanh(x, min_val=-1., max_val=1.)