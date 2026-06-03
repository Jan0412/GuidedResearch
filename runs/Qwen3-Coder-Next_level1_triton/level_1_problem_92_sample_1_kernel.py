import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_elements,  # Total number of elements
    stride,  # Stride along the dimension we're computing cumsum
    dim_size,  # Size of the dimension we're computing cumsum
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one "row" along the cumsum dimension
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * stride
    
    # Initialize running sum to 0 for exclusive cumsum
    running_sum = 0.0
    
    # Process each element in the dimension
    for i in range(dim_size):
        # Calculate current element offset
        elem_offset = row_start + i * stride
        
        # Load current element
        curr_val = tl.load(x_ptr + elem_offset)
        
        # Store the running sum (exclusive - before adding current element)
        tl.store(out_ptr + elem_offset, running_sum)
        
        # Update running sum with current element
        running_sum = running_sum + curr_val


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    """
    Compute exclusive cumulative sum along specified dimension using Triton.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute exclusive cumsum
        
    Returns:
        Tensor with exclusive cumulative sum computed along dim
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and validate dim
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    assert 0 <= dim < len(shape), f"Invalid dimension {dim} for shape {shape}"
    
    # Calculate strides and sizes
    dim_size = shape[dim]
    if dim_size == 0:
        return torch.zeros_like(x)
    
    # Calculate stride along the dimension
    stride = 1
    for i in range(dim + 1, len(shape)):
        stride *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Calculate number of rows (all dimensions except the cumsum dimension)
    num_rows = 1
    for i, s in enumerate(shape):
        if i != dim:
            num_rows *= s
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 128
    
    # Launch kernel - one block per row
    grid = (num_rows,)
    
    exclusive_cumsum_kernel[grid](
        x, out, 
        x.numel(), 
        stride, 
        dim_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum (does not include the current element)
    using Triton kernels.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)