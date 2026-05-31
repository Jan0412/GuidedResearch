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
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ELU implementation: f(x) = alpha * (exp(x) - 1) if x < 0 else x
    exp_x = tl.exp(x)
    elu_result = alpha * (exp_x - 1.0)
    out = tl.where(x < 0.0, elu_result, x)
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_elu(x: torch.Tensor, alpha: float = 1.0):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
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