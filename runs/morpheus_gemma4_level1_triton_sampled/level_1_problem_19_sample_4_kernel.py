import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
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
    # Perform ReLU: max(0, x)
    out = tl.maximum(0.0, x)
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_relu(x: torch.Tensor):
    """
    Triton wrapper for ReLU activation.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    
    # Output tensor
    out = torch.empty_like(x)
    
    # Tuning parameter
    BLOCK_SIZE = 1024
    
    # Grid calculation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Kernel launch
    relu_kernel[grid](
        x, 
        out, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a ReLU activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return triton_relu(x)