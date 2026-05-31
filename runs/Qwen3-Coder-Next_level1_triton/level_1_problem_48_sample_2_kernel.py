import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_cols,  # Number of columns in the reduction dimension
    stride_x,  # Stride along the reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element (i.e., one "row" after removing the reduction dimension)
    row_idx = tl.program_id(0)
    
    # Calculate base pointer for this row
    x_row_start = row_idx * stride_x
    
    # Initialize sum accumulator
    sum_ = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data as float32 for accumulation
        x = tl.load(x_ptr + x_row_start + col_offsets * stride_x, mask=mask, other=0.0)
        sum_ += tl.sum(x.to(tl.float32))
    
    # Compute mean
    mean_val = sum_ / n_cols
    
    # Store result
    tl.store(out_ptr + row_idx, mean_val)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes mean along a specified dimension using Triton kernel.
    
    Args:
        x: Input tensor (will be made contiguous if not already)
        dim: Dimension to reduce over
        
    Returns:
        Output tensor with the specified dimension reduced
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get tensor shape and validate dim
    shape = x.shape
    if dim < 0:
        dim += len(shape)
    assert 0 <= dim < len(shape), f"Dimension {dim} out of range for tensor with {len(shape)} dimensions"
    
    # Compute output shape
    out_shape = list(shape)
    out_shape.pop(dim)
    
    # Calculate parameters for kernel
    # For efficient access, we want to treat the reduction dimension as the innermost dimension
    # Reshape to 2D: (outer_size, reduction_size)
    outer_size = 1
    for i, s in enumerate(shape):
        if i != dim:
            outer_size *= s
    
    reduction_size = shape[dim]
    
    # Create output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # If the tensor is already in the right memory layout, we can compute stride directly
    # Otherwise, we'll reshape to 2D and handle it that way
    if dim == len(shape) - 1:
        # Reduction on last dimension - ideal case
        stride_x = 1
        # Reshape for kernel
        x_2d = x.view(outer_size, reduction_size)
        out_1d = out.view(-1)
    else:
        # Need to move reduction dim to the end for efficiency
        # Create a permutation that moves dim to the end
        perm = list(range(len(shape)))
        perm.append(perm.pop(dim))
        x_permuted = x.permute(perm).contiguous()
        x_2d = x_permuted.view(outer_size, reduction_size)
        out_1d = out.view(-1)
        stride_x = 1  # After permutation and contiguous, last dimension has stride 1
    
    # Determine block size (tunable)
    BLOCK_SIZE = 256
    
    # Grid is 1D: one block per row in the 2D view
    grid = (outer_size,)
    
    # Launch kernel
    mean_kernel[grid](
        x_2d,
        out_1d,
        reduction_size,
        stride_x,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Triton kernel.
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
        Reduces the input tensor along the specified dimension by taking the mean using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean(x, self.dim)