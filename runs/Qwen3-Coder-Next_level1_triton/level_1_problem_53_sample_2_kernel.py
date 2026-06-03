import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch dimension)
    n_cols,  # Number of columns (size of dimension being reduced)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    
    # Initialize minimum to positive infinity
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
    
    # Load and compute partial minimum
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < (n_cols - offset)
        ptr = x_ptr + row_start + offset + cols
        x = tl.load(ptr, mask=mask, other=float('inf'))
        # Element-wise minimum
        min_val = tl.minimum(min_val, x)
    
    # Final reduction within the block to get the minimum
    # Use a tree reduction approach
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if stride < BLOCK_SIZE:
            cols = tl.arange(0, stride)
            mask = cols < stride
            val1 = tl.load(min_val + cols, mask=mask, other=float('inf'))
            val2 = tl.load(min_val + stride + cols, mask=mask, other=float('inf'))
            min_val = tl.minimum(val1, val2)
        stride = stride // 2
    
    # Store the result
    tl.store(out_ptr + row_idx, min_val[0])


def triton_min(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of min reduction over specified dimension.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Tensor with min values along the specified dimension
    """
    # Ensure input is contiguous and on GPU
    x = x.contiguous()
    
    # Get input shape
    shape = x.shape
    ndim = len(shape)
    
    # Normalize dimension to positive
    if dim < 0:
        dim += ndim
    
    # Calculate output shape
    output_shape = list(shape)
    output_shape[dim] = 1
    output_shape = tuple(output_shape)
    
    # Reshape to 2D for easier processing: (batch_size, reduced_dim)
    # Move the target dimension to the last position
    if dim != ndim - 1:
        permute_dims = list(range(ndim))
        permute_dims.pop(dim)
        permute_dims.append(dim)
        x = x.permute(permute_dims)
        shape = x.shape
    
    batch_size = 1
    for s in shape[:-1]:
        batch_size *= s
    reduced_size = shape[-1]
    
    # Prepare output tensor
    out_shape = list(x.shape)
    out_shape[-1] = 1
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    # Launch kernel
    min_reduction_kernel[grid](
        x, out, batch_size, reduced_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Squeeze the dimension that was reduced
    out = out.squeeze(dim=-1)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension
    using custom Triton kernel.
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
        Applies min reduction over the specified dimension to the input tensor
        using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        return triton_min(x, self.dim)