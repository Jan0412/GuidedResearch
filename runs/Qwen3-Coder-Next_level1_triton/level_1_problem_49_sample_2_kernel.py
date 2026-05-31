import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (dim2 in the reduction dimension)
    stride_row,  # Stride between rows
    stride_col,  # Stride between columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate starting pointer for this row
    x_ptr_row = x_ptr + row_idx * stride_row
    out_ptr_row = out_ptr + row_idx
    
    # Initialize max with the smallest possible float value
    max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
    
    # Process the row in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr_row + col_offsets * stride_col, mask=mask)
        
        # Update max
        max_val = tl.maximum(max_val, x)
    
    # Reduce the block max values to a single max value
    # Since we're only using one block per row in this implementation (with BLOCK_SIZE >= n_cols),
    # we can just store the max_val directly
    tl.store(out_ptr_row, max_val)


def triton_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies Max reduction over the specified dimension using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to reduce over.
        
    Returns:
        torch.Tensor: Output tensor after Max reduction over the specified dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shapes and strides
    shape = x.shape
    stride = x.stride()
    
    # Calculate dimensions
    if dim < 0:
        dim = len(shape) + dim
    
    # Get dimensions for the kernel
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    
    n_cols = shape[dim]
    stride_row = stride[dim]
    stride_col = 1
    for i in range(dim + 1, len(shape)):
        stride_row *= shape[i]
        stride_col *= shape[i]
    
    # Prepare output tensor
    out_shape = list(shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Block size - should be >= n_cols to avoid needing reduction between blocks
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Ensure it's at least 128 for good performance, but not larger than needed
    BLOCK_SIZE = min(max(BLOCK_SIZE, 128), 1024)
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the kernel
    max_kernel[grid](
        x, out, n_rows, n_cols, stride_row, stride_col,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Squeeze the reduced dimension to match torch.max behavior
    out = out.squeeze(dim)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Max reduction over a specific dimension using Triton kernel.
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
        Applies Max reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return triton_max(x, self.dim)