import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cumulative_product_kernel(
    input_ptr,
    output_ptr,
    N,
    dim_size,
    stride,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    
    # Load data for this block
    offsets = start_pos + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=1.0)
    
    # Compute cumulative product along the specified dimension
    # For each element, multiply by all previous elements in the same sequence
    result = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    
    # Process elements in reverse order for cumulative product
    for i in range(dim_size):
        # Calculate offset for current element
        current_offset = start_pos + i * stride
        if current_offset < N:
            # Load current value
            current_val = tl.load(input_ptr + current_offset, mask=(current_offset < N), other=1.0)
            # Update result
            result = result * current_val
            
            # Store result at appropriate location
            tl.store(output_ptr + current_offset, result, mask=(current_offset < N))

def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative product along a specific dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    
    # Ensure tensor is contiguous
    x = x.contiguous()
    
    # Get tensor properties
    shape = x.shape
    total_elements = x.numel()
    
    # Handle case where dim is not valid
    if dim < 0:
        dim += len(shape)
    
    # Get size of the specified dimension
    dim_size = shape[dim]
    
    # Calculate stride for the specified dimension
    stride = 1
    for i in range(dim + 1, len(shape)):
        stride *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    cumulative_product_kernel[grid_size](
        x, out, total_elements, dim_size, stride, BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized version of CumulativeProductModel using Triton kernels.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return triton_cumprod(x, self.dim)