import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduce_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch dimension)
    n_cols,  # Number of columns (elements per row to reduce)
    stride_row,  # Stride between rows in the input tensor
    stride_col,  # Stride between elements within a row in the input tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate base pointers for this row
    x_row_start = x_ptr + row_idx * stride_row
    
    # Initialize minimum with a large value
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    
    # Process in chunks of BLOCK_SIZE
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Create offsets for current chunk
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        col_offset = tl.where(mask, offsets, 0)
        x_ptrs = x_row_start + col_offset * stride_col
        x_vals = tl.load(x_ptrs, mask=mask, other=float('inf'))
        
        # Update minimum
        min_val = tl.minimum(min_val, x_vals)
    
    # Reduce the BLOCK_SIZE values to a single minimum
    # Use tree reduction for efficiency
    for i in range(1, BLOCK_SIZE):
        if i < BLOCK_SIZE:
            min_val = tl.minimum(min_val, tl.roll(min_val, i))
    
    # Store the final minimum value for this row
    if tl.program_id(0) < n_rows:
        tl.store(out_ptr + row_idx, min_val[0])


def triton_min_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton-based min reduction over specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Tensor with the minimum values along the specified dimension
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions and strides
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
        
    # For our case, we expect a 3D tensor but we'll handle general case
    # Reshape to 2D if needed: [batch_size, reduced_dim]
    if dim == 1:
        # x is [batch_size, dim1, dim2] and we reduce over dim1
        batch_size = shape[0]
        reduced_size = shape[1]
        remaining_size = shape[2]
        
        # Reshape to [batch_size, reduced_size * remaining_size]
        # But actually, for min reduction over dim1, we want to process
        # each [dim1, dim2] slice for each batch
        # Let's reshape to [batch_size * dim2, dim1]
        x_reshaped = x.permute(0, 2, 1).contiguous().view(batch_size * remaining_size, reduced_size)
        
        # Output will be [batch_size, remaining_size]
        output_shape = shape[:dim] + shape[dim+1:]
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        
        # Kernel parameters
        n_rows = batch_size * remaining_size
        n_cols = reduced_size
        stride_row = x_reshaped.stride(0)
        stride_col = x_reshaped.stride(1)
        
        # Grid: one block per row
        grid = lambda meta: (n_rows,)
        
        # Launch kernel
        min_reduce_kernel[grid](
            x_reshaped, out.view(-1), 
            n_rows, n_cols, 
            stride_row, stride_col,
            BLOCK_SIZE=128
        )
        
        return out
    elif dim == 2:
        # x is [batch_size, dim1, dim2] and we reduce over dim2
        batch_size = shape[0]
        dim1 = shape[1]
        dim2 = shape[2]
        
        # Reshape to [batch_size * dim1, dim2]
        x_reshaped = x.contiguous().view(batch_size * dim1, dim2)
        
        # Output will be [batch_size, dim1]
        output_shape = shape[:dim] + shape[dim+1:]
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        
        # Kernel parameters
        n_rows = batch_size * dim1
        n_cols = dim2
        stride_row = x_reshaped.stride(0)
        stride_col = x_reshaped.stride(1)
        
        # Grid: one block per row
        grid = lambda meta: (n_rows,)
        
        # Launch kernel
        min_reduce_kernel[grid](
            x_reshaped, out.view(-1), 
            n_rows, n_cols, 
            stride_row, stride_col,
            BLOCK_SIZE=128
        )
        
        return out
    else:
        # Fallback to PyTorch for other dimensions
        return torch.min(x, dim=dim)[0]


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton kernels.
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
        Applies min reduction over the specified dimension using Triton kernel.
        """
        return triton_min_reduce(x, self.dim)