import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    x_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    # Offset for the current block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask for boundary conditions
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Softplus(x) = log(1 + exp(x))
    # For numerical stability, we use the identity:
    # softplus(x) = max(0, x) + log(1 + exp(-abs(x)))
    # However, a simpler stable approximation for large x is:
    # if x > 20: return x
    # else: return log(1 + exp(x))
    
    # Using the stable formulation:
    # out = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    # Or the more robust version:
    abs_x = tl.abs(x)
    softplus_val = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    
    # Store result
    tl.store(out_ptr + offsets, softplus_val, mask=mask)

def triton_softplus(x: torch.Tensor):
    """
    Custom Triton implementation of the softplus activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    
    out = torch.empty_like(x)
    
    # Tunable block size
    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    softplus_kernel[grid](
        x, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor via Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)