import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
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
    
    # Numerically stable softplus implementation
    # softplus(x) = log(1 + exp(x))
    # For numerical stability:
    # - If x > 20, exp(x) is very large, so softplus(x) ≈ x
    # - If x < -20, exp(x) is very small, so softplus(x) ≈ exp(x)
    # - Otherwise, compute directly with log1p(exp(x))
    
    exp_x = tl.exp(x)
    # Use tl.where for conditional logic
    cond1 = x > 20.0
    cond2 = x < -20.0
    
    # For large positive x, use x directly
    # For large negative x, use exp(x) 
    # Otherwise, use log1p(exp(x))
    out = tl.where(
        cond1,
        x,
        tl.where(
            cond2,
            exp_x,
            tl.log1p(exp_x)
        )
    )
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor):
    """
    This function wraps the Triton kernel call for Softplus activation.
    
    Args:
        x (torch.Tensor): Input tensor of any shape.
        
    Returns:
        torch.Tensor: Output tensor with Softplus applied, same shape as input.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    softplus_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Softplus activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)