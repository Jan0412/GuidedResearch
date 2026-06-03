import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer (indices)
    n_rows,  # Number of rows (all dimensions except the reduction dimension)
    n_cols,  # Size of reduction dimension
    row_stride,  # Stride to move to next row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for argmax operation.
    Each program processes one row of the input tensor.
    """
    # Get the row index this program handles
    row_idx = tl.program_id(0)
    
    # Calculate starting pointer for this row
    row_start_ptr = x_ptr + row_idx * row_stride
    
    # Initialize max value and index
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    max_idx = tl.arange(0, BLOCK_SIZE)
    
    # Process the row in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        # Calculate offsets
        offsets = start + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid elements
        mask = offsets < n_cols
        
        # Load values
        vals = tl.load(row_start_ptr + offsets, mask=mask, other=-float('inf'))
        # Convert to float32 for computation
        vals = vals.to(tl.float32)
        
        # Update max if current values are larger
        is_greater = vals > max_val
        max_val = tl.where(is_greater, vals, max_val)
        max_idx = tl.where(is_greater, offsets, max_idx)
    
    # Final reduction to find single max across blocks
    # Since Triton doesn't have built-in reduce_max_with_index, we do it sequentially
    # by finding the max in the block and comparing
    block_max_val = max_val[0]
    block_max_idx = max_idx[0]
    
    for i in range(1, BLOCK_SIZE):
        is_greater = max_val[i] > block_max_val
        block_max_val = tl.where(is_greater, max_val[i], block_max_val)
        block_max_idx = tl.where(is_greater, max_idx[i], block_max_idx)
    
    # Store the result
    tl.store(out_ptr + row_idx, block_max_idx)


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton-based argmax implementation.
    
    Args:
        x: Input tensor
        dim: Dimension to perform argmax over
        
    Returns:
        Tensor with argmax indices along the specified dimension
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get input shape
    shape = x.shape
    n_dims = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim = n_dims + dim
    
    # Calculate dimensions for the kernel
    # We'll treat everything as 2D: [n_rows, n_cols] where n_cols is the reduction dimension
    if dim == n_dims - 1:
        # Last dimension - straightforward
        n_rows = 1
        for i in range(n_dims - 1):
            n_rows *= shape[i]
        n_cols = shape[dim]
        row_stride = shape[dim]
    else:
        # Need to reshape to make the reduction dimension last
        # Permute dimensions so reduction dim is last
        dims = list(range(n_dims))
        dims.append(dims.pop(dim))  # Move dim to end
        x_permuted = x.permute(dims)
        shape_permuted = x_permuted.shape
        n_rows = 1
        for i in range(n_dims - 1):
            n_rows *= shape_permuted[i]
        n_cols = shape_permuted[-1]
        row_stride = shape_permuted[-1]
        x = x_permuted.contiguous()
    
    # Ensure BLOCK_SIZE is at least as large as n_cols, or use multiple passes
    BLOCK_SIZE = min(1024, max(32, (n_cols + 31) // 32 * 32))  # Round up to multiple of 32
    
    # Create output tensor
    output_shape = list(shape)
    output_shape.pop(dim)  # Remove the reduction dimension
    out = torch.empty(output_shape, dtype=torch.long, device=x.device)
    
    # Launch kernel
    grid = (n_rows,)
    argmax_kernel[grid](x, out, n_rows, n_cols, row_stride, BLOCK_SIZE=BLOCK_SIZE)
    
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