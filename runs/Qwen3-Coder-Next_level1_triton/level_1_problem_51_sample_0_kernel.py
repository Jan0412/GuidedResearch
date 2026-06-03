import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    x_ptr,                # Input pointer
    out_ptr,              # Output pointer
    n_rows,               # Number of rows to process
    n_cols,               # Number of columns in each row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    
    # Initialize max value and index
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process the row in chunks of BLOCK_SIZE
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual column indices
        col_indices = col_offset + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid indices
        mask = col_indices < n_cols
        
        # Load values
        ptr = x_ptr + row_start + col_indices
        vals = tl.load(ptr, mask=mask, other=-float('inf'))
        
        # Update max if we find a larger value
        is_greater = vals > max_val
        max_val = tl.where(is_greater, vals, max_val)
        max_idx = tl.where(is_greater, col_indices, max_idx)
    
    # Store the result (index of maximum)
    out_ptr[row_idx] = max_idx.to(tl.int64)


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of argmax along a specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute argmax
        
    Returns:
        Tensor with argmax indices along the specified dimension
    """
    # Ensure tensor is contiguous and on CUDA
    x = x.contiguous()
    assert x.is_cuda, "Tensor must be on CUDA device"
    
    # Normalize dimension to positive
    if dim < 0:
        dim = x.dim() + dim
        
    # Get input shape
    shape = x.shape
    n_rows = 1
    n_cols = shape[dim]
    
    # Calculate total elements before and after the dimension
    for i in range(dim):
        n_rows *= shape[i]
        
    # Reshape to 2D for easier processing: (n_rows, n_cols)
    # where n_cols is the size of the dimension we're reducing
    x_2d = x.view(n_rows, n_cols)
    
    # Create output tensor with shape excluding the reduced dimension
    out_shape = list(shape)
    del out_shape[dim]
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    
    # Set kernel parameters
    BLOCK_SIZE = 256  # Tunable parameter
    grid = (n_rows,)  # One block per row
    
    # Launch kernel
    argmax_kernel[grid](x_2d, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension using Triton kernel.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return triton_argmax(x, self.dim)