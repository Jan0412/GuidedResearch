import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (elements in the output)
    n_cols,  # Number of columns (size of reduction dimension)
    stride_row,  # Stride between rows in input tensor
    stride_col,  # Stride between columns in input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one output element)
    row_idx = tl.program_id(0)
    
    # Compute base pointers for this row
    x_row_start = x_ptr + row_idx * stride_row
    out_ptr_row = out_ptr + row_idx
    
    # Initialize max with negative infinity
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    
    # Iterate over columns in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid columns
        mask = col_offsets < n_cols
        
        # Load data (broadcast across rows for this row)
        x_ptrs = x_row_start + col_offsets * stride_col
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        
        # Update maximum
        max_val = tl.maximum(max_val, x)
    
    # Reduce the block maxima to get the final max for this row
    # Since we're using block-level reduction, we need to handle it carefully
    # For simplicity, we'll use a simple approach where each block computes its max
    # and then we do a second pass, but for now let's use the first block's max
    # Actually, let's do a more efficient approach using tl.max with axis=None
    # But Triton doesn't support that directly in a single kernel without synchronization
    
    # Instead, we'll use a tree-reduction approach in a single kernel
    # For simplicity and correctness, we'll use a simple approach with one thread per row
    # Since n_cols can be large (4095), we need to reduce across all columns
    
    # We'll do a sequential reduction in the kernel
    # First initialize with the first valid element
    first_col = tl.arange(0, 1)
    first_mask = first_col < n_cols
    first_ptr = x_row_start + first_col * stride_col
    first_val = tl.load(first_ptr, mask=first_mask, other=-float('inf'))
    max_val = first_val
    
    # Now iterate over remaining columns
    for col_offset in range(1, n_cols):
        col_offset_i = tl.full([1], col_offset, dtype=tl.int32)
        ptr = x_row_start + col_offset_i * stride_col
        val = tl.load(ptr, mask=col_offset_i < n_cols, other=-float('inf'))
        max_val = tl.maximum(max_val, val)
    
    # Store the result
    tl.store(out_ptr_row, max_val)


def triton_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    This function wraps the Triton kernel call for max reduction.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to reduce over
        
    Returns:
        torch.Tensor: Output tensor after max reduction
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get input shape and determine output shape
    shape = list(x.shape)
    n_dims = len(shape)
    
    # Handle negative dimension indexing
    if dim < 0:
        dim += n_dims
    
    # Compute output shape
    output_shape = shape[:dim] + shape[dim+1:]
    n_rows = 1
    for s in output_shape:
        n_rows *= s
    
    # Compute input strides
    strides = list(x.stride())
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Compute block size (tuned for 4095 columns)
    BLOCK_SIZE = 128
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    max_reduce_kernel[grid](
        x, out,
        n_rows, shape[dim],
        strides[0] if dim == 0 else (strides[0] if dim > 0 else strides[dim]),
        strides[dim],
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


# A more efficient version using block-wise reduction for better performance
@triton.jit
def max_reduce_kernel_optimized(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (elements in the output)
    n_cols,  # Number of columns (size of reduction dimension)
    stride_row,  # Stride between rows in input tensor
    stride_col,  # Stride between columns in input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one output element)
    row_idx = tl.program_id(0)
    
    # Compute base pointers for this row
    x_row_start = x_ptr + row_idx * stride_row
    
    # Initialize max with negative infinity
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    
    # Iterate over columns in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid columns
        mask = col_offsets < n_cols
        
        # Load data
        x_ptrs = x_row_start + col_offsets * stride_col
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        
        # Update maximum
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Store the result
    tl.store(out_ptr + row_idx, max_val)


def triton_max_reduce_optimized(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Optimized version of max reduction using block-wise reduction.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to reduce over
        
    Returns:
        torch.Tensor: Output tensor after max reduction
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get input shape and determine output shape
    shape = list(x.shape)
    n_dims = len(shape)
    
    # Handle negative dimension indexing
    if dim < 0:
        dim += n_dims
    
    # Compute output shape
    output_shape = shape[:dim] + shape[dim+1:]
    n_rows = 1
    for s in output_shape:
        n_rows *= s
    
    # Compute input strides
    strides = list(x.stride())
    
    # Create output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Compute block size (tuned for large reduction dimension)
    BLOCK_SIZE = 1024  # Larger block size for better occupancy
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    max_reduce_kernel_optimized[grid](
        x, out,
        n_rows, shape[dim],
        strides[0] if dim == 0 else (strides[0] if dim > 0 else strides[dim]),
        strides[dim],
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension
    using custom Triton kernels.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return triton_max_reduce_optimized(x, self.dim)