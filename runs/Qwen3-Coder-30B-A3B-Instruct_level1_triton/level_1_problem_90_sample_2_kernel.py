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
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Calculate the stride for the dimension we're accumulating over
    # For a tensor of shape [batch_size, dim_size], we accumulate along dim_size
    # So each element in the batch has its own accumulation path
    batch_idx = tl.program_id(1)
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < dim_size
    
    # Load input values for this batch
    input_offsets = batch_idx * dim_size + offsets
    x = tl.load(input_ptr + input_offsets, mask=mask, other=1.0)
    
    # Compute cumulative product
    # We compute the cumulative product from left to right
    cumulative = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    
    # Process elements sequentially to maintain correct order
    for i in range(dim_size):
        # Only process elements within the valid range
        if i < dim_size:
            # Get the current element offset
            current_offset = batch_idx * dim_size + i
            current_val = tl.load(input_ptr + current_offset, mask=(i < dim_size), other=1.0)
            
            # Accumulate the product
            cumulative = cumulative * current_val
            
            # Store the accumulated value
            output_offset = batch_idx * dim_size + i
            tl.store(output_ptr + output_offset, cumulative, mask=(i < dim_size))

def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative product along a specific dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    batch_size = 1
    dim_size = shape[dim]
    
    # Calculate batch size (product of all dimensions except the target dim)
    for i in range(len(shape)):
        if i != dim:
            batch_size *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    
    # Grid configuration
    grid = (
        (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE,  # Number of blocks for dimension
        batch_size  # Number of batches
    )
    
    # Launch the Triton kernel
    cumulative_product_kernel[grid](
        x, 
        out, 
        n_elements, 
        dim_size, 
        batch_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for cumulative product operation.
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
        Forward pass, computing the cumulative product along the specified dimension
        using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return triton_cumprod(x, self.dim)