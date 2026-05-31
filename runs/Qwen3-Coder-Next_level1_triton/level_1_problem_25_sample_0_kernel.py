import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,      # Pointer to input tensor
    out_ptr,    # Pointer to output tensor
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    # Use stable sigmoid computation to avoid overflow
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    
    # Compute swish(x) = x * sigmoid(x)
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Swish activation using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output tensor with Swish applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for FP32
    
    # Grid definition
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Swish activation using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        return triton_swish(x)