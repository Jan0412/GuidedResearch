import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduction_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (elements per row to reduce)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one reduction)
    row_idx = tl.program_id(0)
    
    # Calculate base pointer for this row
    x_row_start = x_ptr + row_idx * n_cols
    out_row_start = out_ptr + row_idx * 1  # output is 1 element per row
    
    # Initialize accumulator for sum
    sum_val = tl.zeros([1], dtype=tl.float32)
    
    # Process in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data with masking
        x_val = tl.load(x_row_start + col_offsets, mask=mask, other=0.0)
        
        # Accumulate sum
        sum_val = sum_val + x_val
    
    # Final reduction: use block-level reduction for better performance
    # But for simplicity and since n_cols can be large, we'll just do sequential reduction here
    # Note: For very large n_cols, a tree reduction would be more efficient
    
    # Store result
    tl.store(out_row_start, sum_val)


def triton_sum_reduction(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs sum reduction over the specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Output tensor with reduced dimension kept (keepdim=True)
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()
    
    # Get input shape and adjust dimension
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    
    # Calculate dimensions for kernel
    # We'll reshape to 2D: [n_rows, n_cols] where n_cols is the dimension to reduce
    if dim == len(shape) - 1:
        # Last dimension: easy case
        n_rows = 1
        for s in shape[:-1]:
            n_rows *= s
        n_cols = shape[dim]
    else:
        # Need to move the target dimension to the end first
        perm = list(range(len(shape)))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(perm).contiguous()
        shape = x.shape
        n_rows = 1
        for s in shape[:-1]:
            n_rows *= s
        n_cols = shape[-1]
    
    # Prepare output tensor
    # The output shape will have the reduced dimension as 1
    out_shape = list(shape)
    out_shape[-1] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Reshape for kernel processing
    x_flat = x.view(n_rows, n_cols)
    out_flat = out.view(n_rows, 1)
    
    # Launch kernel
    BLOCK_SIZE = 256  # Tunable parameter
    grid = lambda meta: (n_rows,)
    
    sum_reduction_kernel[grid](x_flat, out_flat, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs sum reduction over a specified dimension using Triton kernel.
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