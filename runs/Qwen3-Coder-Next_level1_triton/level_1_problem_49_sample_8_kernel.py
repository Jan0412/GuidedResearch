import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduction_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns to reduce over
    stride_row,  # Stride between rows
    stride_col,  # Stride between columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element (one row after reduction)
    row_idx = tl.program_id(0)
    
    # Pointer to the start of this row
    x_row_ptr = x_ptr + row_idx * stride_row
    
    # Initialize max with negative infinity
    max_val = -float('inf')
    
    # Process in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x_vals = tl.load(
            x_row_ptr + col_offsets * stride_col,
            mask=mask,
            other=-float('inf')
        )
        
        # Compute max
        max_val = tl.maximum(max_val, tl.max(x_vals))
    
    # Store result
    tl.store(out_ptr + row_idx, max_val)


def triton_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Custom Triton implementation of max reduction over a specified dimension.
    Optimized for FP32 precision.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor shape and dimensions
    shape = x.shape
    ndim = len(shape)
    
    # Normalize dimension to positive
    if dim < 0:
        dim += ndim
    
    # Calculate output shape
    out_shape = list(shape)
    out_shape[dim] = 1
    out_shape = tuple(out_shape)
    
    # Reshape to 2D for easier processing: (batch_size, reduced_dim)
    # Move the reduction dimension to the last position
    if dim != ndim - 1:
        x_permuted = x.permute(*[i for i in range(ndim) if i != dim], dim)
    else:
        x_permuted = x
    
    # Reshape to 2D: [n_rows, n_cols]
    n_rows = 1
    for i in range(ndim - 1):
        n_rows *= x_permuted.shape[i]
    n_cols = x_permuted.shape[-1]
    
    x_2d = x_permuted.reshape(n_rows, n_cols)
    
    # Prepare output tensor
    out = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256  # Tunable parameter
    grid = (n_rows,)
    
    # Calculate strides
    stride_row = x_2d.stride(0)
    stride_col = x_2d.stride(1)
    
    # Launch kernel
    max_reduction_kernel[grid](
        x_2d,
        out,
        n_rows,
        n_cols,
        stride_row,
        stride_col,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape output to match expected output shape
    out = out.view(*out_shape)
    
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