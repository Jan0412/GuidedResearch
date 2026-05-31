import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def elu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ELU: f(x) = alpha * (exp(x) - 1) if x < 0 else x
    # We compute exp(x) - 1 efficiently using tl.expm1 for better numerical stability
    condition = x < 0.0
    exp_x_minus_1 = tl.where(condition, tl.expm1(x), 0.0)
    out = tl.where(condition, alpha * exp_x_minus_1, x)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_elu(x: torch.Tensor, alpha: float):
    """
    This function wraps the Triton kernel call for ELU activation.
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
    elu_kernel[grid](x, out, n_elements, alpha, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs ELU activation using Triton kernel.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        return triton_elu(x, self.alpha)