import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
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
    
    # Compute softplus using numerically stable formulation:
    # softplus(x) = log(1 + exp(beta * x))
    # For numerical stability:
    # - If beta * x > threshold, use beta * x (since exp(beta*x) dominates)
    # - Otherwise, use log(1 + exp(beta * x))
    
    # Compute beta * x
    beta_x = beta * x
    
    # Create mask for values above threshold
    above_threshold = beta_x > threshold
    
    # For values above threshold: use beta * x (approximation)
    # For values below threshold: use log(1 + exp(beta * x))
    # To compute log(1 + exp(z)) in a stable way:
    # if z < threshold: log(1 + exp(z))
    # else: z + log(1 + exp(-z))
    
    # We'll use the stable approach:
    # if z > threshold: return z
    # else: return log(1 + exp(z))
    # This is already stable because we set threshold to something like 20
    
    # For z <= threshold, compute log(1 + exp(z))
    # Use exp(z) directly since z is not too large
    exp_beta_x = tl.exp(beta_x)
    result = tl.where(
        above_threshold,
        beta_x,  # If above threshold, use approximation
        tl.log1p(exp_beta_x)  # Otherwise compute log(1 + exp(z))
    )
    
    # Scale by 1/beta
    result = result / beta
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_softplus(x: torch.Tensor, beta: float = 1.0):
    """
    This function wraps the Triton kernel call for Softplus activation.
    It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
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
    
    # Set threshold for numerical stability (typically 20 for float32)
    threshold_val = 20.0
    
    # Launch the Triton kernel
    softplus_kernel[grid](x, out, n_elements, beta=beta, threshold=threshold_val, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using Triton kernel.
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