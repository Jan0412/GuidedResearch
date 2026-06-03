import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor (indices)
    n_rows,  # Number of rows (outer dimension)
    n_cols,  # Number of columns (inner dimension to reduce)
    stride_row,  # Stride between rows
    stride_col,  # Stride between columns in a row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Initialize min value and index
    min_val = tl.float32(tl.cdiv(1, 0))  # -inf
    min_idx = tl.zeros([1], dtype=tl.int32)
    
    # Start pointer for this row
    row_start = x_ptr + row_idx * stride_row
    
    # Process in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data
        ptr = row_start + offsets * stride_col
        val = tl.load(ptr, mask=mask, other=float('inf'))
        
        # Find minimum in this block
        block_min_idx = tl.argmin(val, axis=0)
        block_min_val = tl.load(ptr + block_min_idx * stride_col)
        
        # Update global minimum
        is_new_min = block_min_val < min_val
        min_val = tl.where(is_new_min, block_min_val, min_val)
        min_idx = tl.where(is_new_min, block_min_idx + start_col, min_idx)
    
    # Store result
    out_ptr[row_idx] = min_idx


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Custom Triton implementation of argmin along specified dimension.
    Optimized for FP32 tensors.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    if dim < 0:
        dim += len(shape)
    
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
    
    stride_col = 1 if dim == len(shape) - 1 else shape[dim + 1]
    for i in range(dim + 1, len(shape)):
        stride_col *= shape[i]
    
    # Configure kernel launch
    BLOCK_SIZE = triton.next_power_of_2(n_cols) if n_cols > 0 else 1
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Cap at 1024 for efficiency
    
    # Launch kernel
    grid = (n_rows,)
    argmin_kernel[grid](
        x, out,
        n_rows, n_cols,
        stride_row, stride_col,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that finds the index of the minimum value along a specified dimension
    using custom Triton kernel instead of PyTorch's built-in argmin.
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