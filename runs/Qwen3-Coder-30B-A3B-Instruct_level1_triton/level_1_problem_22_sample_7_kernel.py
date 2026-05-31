import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute tanh using approximation for better performance
    # Using the identity: tanh(x) = 2 * sigmoid(2x) - 1
    # But we'll use a more direct approach with fused operations
    x = x * 2.0  # Scale for sigmoid computation
    exp_x = tl.exp(x)
    sigmoid_x = exp_x / (exp_x + 1.0)
    tanh_x = 2.0 * sigmoid_x - 1.0
    
    # Store the result
    tl.store(output_ptr + offsets, tanh_x, mask=mask)

def triton_tanh(x: torch.Tensor):
    """
    Triton implementation of tanh activation
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tuned for good performance on modern GPUs
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Tanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        return triton_tanh(x)