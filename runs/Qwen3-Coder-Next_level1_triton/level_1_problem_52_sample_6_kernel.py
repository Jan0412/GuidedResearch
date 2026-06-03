import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output indices pointer
    n_rows,  # Number of rows to process
    n_cols,  # Number of columns in each row
    stride_row,  # Stride between rows
    stride_col,  # Stride between columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate the starting pointer for this row
    x_row_start = x_ptr + row_idx * stride_row
    out_row_start = out_ptr + row_idx
    
    # Initialize minimum value and index
    min_val = tl.full((1,), float('inf'), dtype=tl.float32)
    min_idx = tl.zeros((1,), dtype=tl.int32)
    
    # Process columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        cols = start_col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        # Load values
        offsets = x_row_start + cols * stride_col
        vals = tl.load(offsets, mask=mask, other=float('inf'))
        
        # Compare and update minimum
        # For FP32, we need to handle NaNs properly, but using tl.minimum
        curr_min = tl.minimum(min_val, vals)
        
        # Update min_val and min_idx
        is_new_min = (vals < min_val) | (mask & (min_val == float('inf')))
        min_val = tl.where(is_new_min, vals, min_val)
        min_idx = tl.where(is_new_min, cols, min_idx)
    
    # Store result
    tl.store(out_row_start, min_idx)


class TritonArgminFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Determine dimensions for the kernel
        if dim < 0:
            dim = x.dim() + dim
            
        # Get shape info
        shape = x.shape
        n_rows = 1
        for i in range(dim):
            n_rows *= shape[i]
        
        n_cols = shape[dim]
        
        # Prepare output tensor
        out_shape = list(shape)
        out_shape[dim] = 1
        out = torch.empty(out_shape, dtype=torch.long, device=x.device)
        
        # Calculate strides
        stride_row = 1
        for i in range(dim + 1, len(shape)):
            stride_row *= shape[i]
        stride_col = 1
        
        # Set block size
        BLOCK_SIZE = 128
        
        # Grid: one block per row
        grid = (n_rows,)
        
        # Launch kernel
        argmin_kernel[grid](
            x, out, n_rows, n_cols,
            stride_row, stride_col,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    return TritonArgminFunction.apply(x, dim)


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension
    using custom Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return triton_argmin(x, self.dim)