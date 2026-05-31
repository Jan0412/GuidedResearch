import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardsigmoid_kernel(
    x_ptr,      # Pointer to input tensor
    out_ptr,    # Pointer to output tensor
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the offset for the current program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load the input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute HardSigmoid: clamp((x + 3.0) / 6.0, 0.0, 1.0)
    # We multiply by (1/6) instead of dividing by 6 for potential performance gains
    res = (x + 3.0) * (1.0 / 6.0)
    res = tl.maximum(0.0, tl.minimum(1.0, res))
    
    # Store the result back to the output tensor
    tl.store(out_ptr + offsets, res, mask=mask)

def triton_hardsigmoid(x: torch.Tensor):
    """
    Wrapper function to launch the Triton HardSigmoid kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure the tensor is contiguous for linear memory access
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    # Block size can be tuned; 1024 is generally a good default for element-wise ops
    BLOCK_SIZE = 1024
    
    # Define the grid: one block for every BLOCK_SIZE elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a HardSigmoid activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor using the Triton implementation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        return triton_hardsigmoid(x)