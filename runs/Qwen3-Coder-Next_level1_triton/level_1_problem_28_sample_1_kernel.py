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
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute HardSigmoid: max(0, min(1, (x + 3) / 6))
    # HardSigmoid formula: relu6(x + 3) / 6
    # Where relu6(x) = min(max(x, 0), 6)
    temp = (x + 3.0) / 6.0
    out = tl.minimum(tl.maximum(temp, 0.0), 1.0)
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_hardsigmoid(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for HardSigmoid activation.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output tensor with HardSigmoid applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
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
        Applies HardSigmoid activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return triton_hardsigmoid(x)