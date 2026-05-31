import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cumulative_product_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=1.0)
    
    # Initialize accumulator for cumulative product
    acc = 1.0
    
    # Compute cumulative product along the specified dimension
    # For each element, multiply by all previous elements in the sequence
    for i in range(dim_size):
        # Calculate offset for current element in the dimension
        current_offset = (offsets // stride) * stride + i
        # Ensure we don't go out of bounds
        current_mask = (current_offset < n_elements) & mask
        
        # Load current value
        current_val = tl.load(input_ptr + current_offset, mask=current_mask, other=1.0)
        
        # Update accumulator
        acc = acc * current_val
        
        # Store result
        tl.store(output_ptr + current_offset, acc, mask=current_mask)

def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative product along a specific dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Calculate total elements and dimension size
    n_elements = x.numel()
    dim_size = x.shape[dim]
    
    # Calculate stride for the specified dimension
    stride = 1
    for i in range(dim + 1, len(x.shape)):
        stride *= x.shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine block size and grid
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    cumulative_product_kernel[grid](
        x, out, n_elements, dim_size, stride, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)