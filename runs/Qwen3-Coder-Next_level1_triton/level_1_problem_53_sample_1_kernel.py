import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch dimension)
    n_cols,  # Number of columns (size of dimension being reduced)
    stride_row,  # Stride between rows
    stride_col,  # Stride between columns in the reduced dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one min computation)
    row_idx = tl.program_id(0)
    
    # Compute base pointer for this row
    row_start = row_idx * stride_row
    
    # Initialize minimum with large value
    min_val = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    
    # Process in chunks of BLOCK_SIZE
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        # Create offsets for this chunk
        offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Compute pointer to this element
        ptr = row_start + offsets * stride_col
        
        # Load values, handling out-of-bounds
        vals = tl.load(ptr, mask=mask, other=float('inf'))
        
        # Update minimum
        min_val = tl.minimum(min_val, vals)
    
    # Now do reduction within the block to find the final minimum
    # We'll use a tree reduction approach
    block_size = BLOCK_SIZE
    while block_size > 1:
        half = block_size // 2
        # Only process first half of remaining elements
        if tl.program_id(0) == 0 and tl.arange(0, half) < block_size:
            # Create masks for current half and second half
            left_mask = tl.arange(0, half) < block_size
            right_mask = (tl.arange(0, half) + half) < block_size
            
            left_vals = tl.load(min_val + tl.arange(0, half), mask=left_mask)
            right_vals = tl.load(min_val + tl.arange(0, half) + half, mask=right_mask)
            
            # Compute minimum
            merged = tl.minimum(left_vals, right_vals)
            tl.store(min_val + tl.arange(0, half), merged, mask=left_mask)
        
        block_size = half
    
    # Store the final result (only the first thread in each block writes)
    if tl.program_id(0) == 0:
        final_min = tl.load(min_val)
        tl.store(out_ptr + row_idx, final_min)


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    This function wraps the Triton kernel call for min reduction.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension to reduce over
        
    Returns:
        torch.Tensor: Output tensor after min reduction
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get input shape and ensure dim is valid
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    
    # Compute output shape
    out_shape = list(shape)
    out_shape[dim] = 1
    out_shape = tuple(out_shape)
    
    # Reshape to 2D for easier processing: [batch_size, reduced_dim]
    # Move the reduction dimension to the last position
    if dim != len(shape) - 1:
        perm = list(range(len(shape)))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(perm)
    
    # Get dimensions after reshaping
    n_rows = 1
    for i in range(len(shape) - 1):
        n_rows *= shape[i]
    n_cols = shape[dim]
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = min(128, triton.next_power_of_2(n_cols))
    
    # Determine grid size
    grid = (n_rows,)
    
    # Compute strides
    stride_row = x.stride(-1) if len(shape) == 1 else x.stride()[-2] if len(shape) > 1 else 1
    stride_col = x.stride(-1)
    
    # Launch the kernel
    min_reduction_kernel[grid](
        x, out, n_rows, n_cols, stride_row, stride_col, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension using Triton kernel.
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
        Applies min reduction over the specified dimension to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min(x, self.dim)