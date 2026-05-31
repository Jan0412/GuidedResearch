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
    # Get the block ID
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=1.0)
    
    # Initialize output tensor
    output = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    
    # Perform cumulative product along the specified dimension
    for i in range(dim_size):
        # Calculate the position in the tensor
        pos = i * stride + offsets
        # Ensure we don't go out of bounds
        current_mask = (pos < n_elements) & mask
        # Update output with cumulative product
        output = output * tl.load(input_ptr + pos, mask=current_mask, other=1.0)
        # Store intermediate results if needed
        tl.store(output_ptr + pos, output, mask=current_mask)

def triton_cumulative_product(x: torch.Tensor, dim: int):
    """
    This function wraps the Triton kernel call for cumulative product.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor properties
    shape = x.shape
    total_elements = x.numel()
    
    # Calculate dimension size and stride
    dim_size = shape[dim]
    stride = 1
    for i in range(dim + 1, len(shape)):
        stride *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = total_elements
    
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    cumulative_product_kernel[grid](
        x, 
        out, 
        n_elements, 
        dim_size, 
        stride, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.
    Optimized with Triton kernels.
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
        return triton_cumulative_product(x, self.dim)