import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,      # Pointer to input
    out_ptr,    # Pointer to output
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the offset for the current program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Triton's tl.exp handles FP32 precision
    out = 1.0 / (1.0 + tl.exp(-x))
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_sigmoid(x: torch.Tensor):
    """
    Triton wrapper for the sigmoid activation function.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure input is contiguous for pointer arithmetic
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    # BLOCK_SIZE can be tuned; 1024 is generally a good default for element-wise kernels
    BLOCK_SIZE = 1024
    
    # Grid is 1D since the operation is element-wise across the flattened tensor
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Sigmoid activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        return triton_sigmoid(x)