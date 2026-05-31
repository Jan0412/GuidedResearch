import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def selu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID and offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SELU activation: scale * (alpha * exp(x) - alpha) if x <= 0, else scale * x
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    
    # Compute SELU using conditional expression
    result = tl.where(
        x > 0,
        scale * x,
        scale * (alpha * tl.exp(x) - alpha)
    )
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_selu(x: torch.Tensor):
    """
    Applies SELU activation using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor
    
    Returns:
        torch.Tensor: Output tensor with SELU applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 512  # Optimized block size for large tensors
    
    # Grid configuration
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    selu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs SELU activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
        
        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return triton_selu(x)