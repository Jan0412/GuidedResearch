import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    n_rows,         # Number of rows (non-reduced dimensions)
    n_cols,         # Number of columns (reduction dimension)
    stride_row,     # Stride between rows
    stride_col,     # Stride between elements in a row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the reduction
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    # Pointer to the start of the current row
    row_ptr = x_ptr + row_idx * stride_row

    # Accumulator for the sum
    acc = 0.0
    
    # Iterate over the reduction dimension in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        # Load elements of the row
        vals = tl.load(row_ptr + cols * stride_col, mask=mask, other=0.0)
        # Sum the elements in the block
        acc += tl.sum(vals)

    # Store the mean (sum / number of elements)
    tl.store(out_ptr + row_idx, acc / n_cols)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton-optimized mean reduction along a specific dimension.
    """
    # Ensure the tensor is on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA."
    
    original_shape = x.shape
    ndim = x.ndim
    # Normalize dimension to be positive
    if dim < 0:
        dim += ndim
    
    n_red = original_shape[dim]
    
    # 1. Permute the tensor to move the reduction dimension to the last position
    # This allows us to treat the reduction as a row-wise operation.
    dims = list(range(ndim))
    dims.pop(dim)
    dims.append(dim)
    x_transposed = x.permute(*dims)
    
    # 2. Flatten the non-reduced dimensions into a single row dimension
    # reshape() might return a view or a copy, but we use strides to stay generic.
    x_flat = x_transposed.reshape(-1, n_red)
    n_rows = x_flat.shape[0]
    
    # Get the actual strides of the flattened view
    stride_row = x_flat.stride(0)
    stride_col = x_flat.stride(1)
    
    # Prepare the output tensor
    out = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    
    # Grid: one program per row
    grid = (n_rows,)
    
    # Block size for the reduction loop
    BLOCK_SIZE = 1024
    
    # Launch the kernel
    mean_kernel[grid](
        x_flat, 
        out, 
        n_rows, 
        n_red, 
        stride_row, 
        stride_col, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 3. Reshape the output back to the expected shape (original shape without the reduced dim)
    out_shape = [original_shape[i] for i in range(ndim) if i != dim]
    return out.view(*out_shape)


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Triton.
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
        Reduces the input tensor along the specified dimension by taking the mean.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean(x, self.dim)