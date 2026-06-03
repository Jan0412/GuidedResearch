import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    dim,
    stride_before_dim,
    stride_dim,
    stride_after_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the starting index for each block in the dimension we're processing
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Compute the offset within the dimension
    offset_in_dim = block_start % dim_size
    # Compute which "slice" we're in (all dimensions except the target dim)
    slice_idx = block_start // dim_size
    
    # Calculate the actual starting pointer for this block
    # We need to map from the flattened index to the actual tensor coordinates
    before_idx = slice_idx // stride_after_dim
    after_idx = slice_idx % stride_after_dim
    
    # Start from the end of the dimension for reverse cumsum
    start_idx = dim_size - 1 - offset_in_dim
    
    # Compute the actual pointer for this position
    ptr = x_ptr + before_idx * stride_before_dim + start_idx * stride_dim + after_idx * stride_after_dim
    
    # Accumulator for the reverse cumulative sum
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process elements in reverse order within the dimension
    for i in range(dim_size):
        # Load current element
        val = tl.load(ptr, mask=offset_in_dim < dim_size, other=0.0)
        acc += val.to(tl.float32)
        tl.store(out_ptr + ptr - x_ptr, acc.to(x_ptr.dtype.element_ty), mask=offset_in_dim < dim_size)
        
        # Move to the next element in the dimension (forward direction)
        ptr -= stride_dim
        offset_in_dim = (offset_in_dim + 1) % dim_size


def triton_reverse_cumsum(x: torch.Tensor, dim: int):
    """
    Compute reverse cumulative sum along specified dimension using Triton kernel.
    
    Reverse cumsum: for [a, b, c, d], returns [a+b+c+d, b+c+d, c+d, d]
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor dimensions and strides
    shape = x.shape
    strides = x.stride()
    
    # Calculate strides for the dimension operations
    dim_size = shape[dim]
    stride_dim = strides[dim]
    
    # Calculate stride_before_dim and stride_after_dim
    stride_before_dim = 1
    for i in range(dim):
        stride_before_dim *= shape[i]
    
    stride_after_dim = 1
    for i in range(dim + 1, len(shape)):
        stride_after_dim *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Total number of elements
    n_elements = x.numel()
    
    # Block size for parallelization
    BLOCK_SIZE = 256
    
    # Grid configuration - we process one block per element in the dimension
    # but we need to handle the entire tensor
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    reverse_cumsum_kernel[grid](
        x, out, n_elements, dim_size, dim,
        stride_before_dim, stride_dim, stride_after_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a reverse cumulative sum operation along a specified dimension.
    
    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use our optimized Triton kernel for reverse cumulative sum
        return triton_reverse_cumsum(x, self.dim)