import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    beta: tl.constexpr,
    threshold: tl.constexpr,
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
    
    # Compute softplus: softplus(x) = log(1 + exp(beta * x))
    # Use numerical stable implementation:
    # if beta * x > threshold, then softplus(x) ~ beta * x
    # else: softplus(x) = log(1 + exp(beta * x))
    beta_x = beta * x
    out = tl.where(
        beta_x > threshold,
        beta_x,
        tl.log(1.0 + tl.exp(beta_x))
    )
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0):
    """
    This function wraps the Triton kernel call for Softplus activation.
    
    Args:
        x: Input tensor
        beta: Softplus beta parameter (default: 1.0)
        threshold: Threshold for numerical stability (default: 20.0)
        
    Returns:
        Output tensor with Softplus applied
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
    softplus_kernel[grid](x, out, n_elements, beta=beta, threshold=threshold, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Softplus activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)