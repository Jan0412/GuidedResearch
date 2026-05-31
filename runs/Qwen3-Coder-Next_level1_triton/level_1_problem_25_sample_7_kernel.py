import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute global offset for this program
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    # Use exp2 for better performance: sigmoid(x) = 1 / (1 + exp2(-x * log(2)))
    log2 = 0.6931471805599453  # ln(2)
    exp_arg = -x * log2
    exp_val = tl.exp2(exp_arg)
    sigmoid_x = 1.0 / (1.0 + exp_val)
    
    # Compute swish(x) = x * sigmoid(x)
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x: torch.Tensor):
    """
    Triton implementation of Swish activation function.
    
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
    BLOCK_SIZE = 512  # Optimized block size for FP32 operations
    
    # Grid definition
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Swish activation using Triton kernel.
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