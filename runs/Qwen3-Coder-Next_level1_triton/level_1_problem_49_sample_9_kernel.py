import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduction_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate starting pointer for this row
    x_ptr += row_idx * n_cols
    out_ptr += row_idx
    
    # Initialize max with negative infinity
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    
    # Process columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + col_offsets, mask=mask, other=-float('inf'))
        
        # Update max
        max_val = tl.maximum(max_val, x.max())
    
    # Store the result
    tl.store(out_ptr, max_val)


def triton_max_reduction(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs max reduction over specified dimension using Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = list(x.shape)
    n_rows = 1
    for i, s in enumerate(shape):
        if i != dim:
            n_rows *= s
    n_cols = shape[dim]
    
    # Prepare output shape
    output_shape = shape[:dim] + shape[dim+1:]
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 256
    
    # Determine grid size (one block per row)
    grid = (n_rows,)
    
    # Launch kernel
    max_reduction_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
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
        return triton_max_reduction(x, self.dim)