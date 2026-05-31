import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't access out-of-bounds memory
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute Tanh activation
    out = tl.tanh(x)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_tanh(x: torch.Tensor):
    """
    Wrapper function for the Triton tanh kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure the tensor is contiguous for linear memory access
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    # Block size of 1024 is generally efficient for element-wise operations
    BLOCK_SIZE = 1024
    
    # Calculate the grid: one dimension for the total number of elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs a Tanh activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor on CUDA.

        Returns:
            torch.Tensor: Output tensor with Tanh applied.
        """
        return triton_tanh(x)