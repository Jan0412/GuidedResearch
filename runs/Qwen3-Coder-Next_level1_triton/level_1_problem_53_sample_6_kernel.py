import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns per row (dim1 * dim2 for dimension reduction)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize minimum with large value
    min_val = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    
    # Process the row in chunks of BLOCK_SIZE
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + offsets
        mask = col_offsets < n_cols
        # Load data with masking
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=float("inf"))
        # Update minimum
        min_val = tl.minimum(min_val, x)
    
    # Perform reduction within the block
    for i in range(BLOCK_SIZE // 2):
        if i < BLOCK_SIZE // 2:
            min_val = tl.minimum(min_val[:BLOCK_SIZE//2], min_val[BLOCK_SIZE//2:])
    
    # Final reduction to single value
    if BLOCK_SIZE >= 32:
        min_val = tl.minimum(min_val[:16], min_val[16:32])
    if BLOCK_SIZE >= 16:
        min_val = tl.minimum(min_val[:8], min_val[8:16])
    if BLOCK_SIZE >= 8:
        min_val = tl.minimum(min_val[:4], min_val[4:8])
    if BLOCK_SIZE >= 4:
        min_val = tl.minimum(min_val[:2], min_val[2:4])
    if BLOCK_SIZE >= 2:
        min_val = tl.minimum(min_val[:1], min_val[1:2])
    
    # Store the result
    tl.store(out_ptr + row_idx, min_val[0])


def triton_min_reduction(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton-based min reduction over specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Tensor with min values along specified dimension
    """
    # Ensure input is contiguous and on CUDA
    x = x.contiguous()
    
    # Get input shape
    shape = x.shape
    ndim = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim += ndim
    
    # Calculate sizes
    outer_size = 1
    reduce_size = shape[dim]
    inner_size = 1
    
    for i in range(dim):
        outer_size *= shape[i]
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    
    # Reshape to 2D: [outer_size, reduce_size * inner_size]
    # For our case with batch_size=128, dim1=4096, dim2=4095
    # if dim=1, then shape is [128, 4096, 4095], we reshape to [128, 4096*4095]
    x_reshaped = x.view(outer_size, reduce_size * inner_size)
    
    # Output shape: all dimensions except dim
    output_shape = list(shape)
    del output_shape[dim]
    
    # Allocate output tensor
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Configure kernel parameters
    n_rows = outer_size
    n_cols = reduce_size * inner_size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    
    # Launch kernel
    grid = (n_rows,)
    min_reduction_kernel[grid](x_reshaped, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for min reduction.
    """
    def __init__(self, dim: int):
        """
        Initializes the optimized model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min_reduction(x, self.dim)