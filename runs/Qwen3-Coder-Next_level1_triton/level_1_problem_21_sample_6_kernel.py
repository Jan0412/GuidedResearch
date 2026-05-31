import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
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
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Using exp(-x) for better numerical stability with large positive x
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    out = 1.0 / (1.0 + exp_neg_x)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of sigmoid activation.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with sigmoid applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Common block size for element-wise operations
    
    # Grid calculation
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for sigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-optimized Sigmoid activation to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return triton_sigmoid(x)