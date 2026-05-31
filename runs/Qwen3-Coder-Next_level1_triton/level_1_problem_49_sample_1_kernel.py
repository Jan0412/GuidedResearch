import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (elements per row to reduce)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Initialize max with the smallest possible float value
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    
    # Iterate over columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data with masking
        data = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        
        # Update max
        max_val = tl.maximum(max_val, data)
    
    # Store the result
    tl.store(out_ptr + row_idx, max_val)


def triton_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs max reduction over the specified dimension using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to reduce over
        
    Returns:
        torch.Tensor: Output tensor after max reduction
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get input shape and adjust dimension
    shape = x.shape
    if dim < 0:
        dim += len(shape)
    
    # Calculate dimensions for kernel
    # We want to reduce over the specified dimension, so we treat it as columns
    # and all other dimensions as batch dimensions
    n_rows = 1
    for i, s in enumerate(shape):
        if i != dim:
            n_rows *= s
    n_cols = shape[dim]
    
    # Prepare output shape
    out_shape = list(shape)
    out_shape[dim] = 1
    out_shape = tuple(out_shape)
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    max_reduction_kernel[grid](
        x, 
        out, 
        n_rows, 
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
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