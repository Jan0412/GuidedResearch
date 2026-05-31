import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduction_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride,
    col_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one row
    row_idx = tl.program_id(0)
    
    # Calculate starting pointer for this row
    row_start_ptr = x_ptr + row_idx * row_stride
    
    # Accumulator for the sum
    sum_acc = tl.zeros([1], dtype=tl.float32)
    
    # Process in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        ptr = row_start_ptr + col_offsets * col_stride
        x = tl.load(ptr, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_acc += tl.sum(x, axis=0)
    
    # Store result
    out_ptr_row = out_ptr + row_idx
    tl.store(out_ptr_row, sum_acc)


def triton_sum_reduction(x: torch.Tensor, dim: int):
    """
    Performs sum reduction along the specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Output tensor with the specified dimension reduced
    """
    # Ensure input is contiguous and on GPU
    x = x.contiguous()
    
    # Get shape info
    shape = x.shape
    ndim = len(shape)
    
    # Normalize negative dimension indices
    if dim < 0:
        dim = ndim + dim
    
    # Calculate strides for efficient memory access
    # We want to optimize for the case where we're reducing along dimension 'dim'
    # and reading sequentially along that dimension
    
    # For simplicity, we'll reshape to 2D: (all_other_dims, reduce_dim)
    # and then handle the reduction
    
    if ndim == 1:
        # Special case: 1D tensor
        n_rows = 1
        n_cols = shape[0]
        row_stride = 1
        col_stride = 1
    else:
        # Reshape to 2D for easier handling
        # Move the reduction dimension to the last position
        perm = list(range(ndim))
        perm.pop(dim)
        perm.append(dim)
        x_perm = x.permute(perm)
        
        # Now reshape to 2D: (product_of_all_except_last, last_dim)
        n_rows = 1
        for i in range(ndim - 1):
            n_rows *= x_perm.shape[i]
        n_cols = x_perm.shape[-1]
        
        # Calculate strides - for contiguous tensor, strides are in bytes
        # We need to convert to element counts
        element_size = x.element_size()
        row_stride = x_perm.stride(-2) * element_size if ndim > 1 else element_size
        col_stride = x_perm.stride(-1) * element_size
    
    # Prepare output tensor
    # The output shape should have the reduced dimension set to 1
    out_shape = list(shape)
    out_shape[dim] = 1
    
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Determine grid size (number of rows to process in parallel)
    grid = (n_rows,)
    
    # Launch kernel
    sum_reduction_kernel[grid](
        x_ptr=x_perm.data_ptr() if ndim > 1 else x.data_ptr(),
        out_ptr=out.data_ptr(),
        n_rows=n_rows,
        n_cols=n_cols,
        row_stride=row_stride,
        col_stride=col_stride,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension
    using custom Triton kernels.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        return triton_sum_reduction(x, self.dim)