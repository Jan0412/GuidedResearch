import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor (indices)
    n_rows,  # Number of rows (batch dimension)
    n_cols,  # Number of columns (dimension to reduce)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_start = row_idx * n_cols
    
    # Initialize max value and index
    max_val = tl.full([1], float("-inf"), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Iterate over the row in chunks
    for start in range(0, n_cols, BLOCK_SIZE):
        # Compute offsets for this block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        # Create mask for valid elements
        mask = offsets < n_cols
        
        # Load values (using tl.where to handle masked positions)
        vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=float("-inf"))
        
        # Compare with current max
        is_greater = vals > max_val
        max_val = tl.where(is_greater, vals, max_val)
        max_idx = tl.where(is_greater, offsets, max_idx)
    
    # Store the result
    tl.store(out_ptr + row_idx, max_idx)


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton implementation of argmax.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    dim_size = shape[dim]
    other_dims = [i for i in range(len(shape)) if i != dim]
    
    # Reshape to 2D: (batch_size, feature_size)
    # where feature_size = dim_size
    # We need to move the target dimension to the last position first
    perm = other_dims + [dim]
    x_permuted = x.permute(perm)
    new_shape = x_permuted.shape
    batch_size = 1
    for s in new_shape[:-1]:
        batch_size *= s
    feature_size = new_shape[-1]
    
    # Reshape to 2D for processing
    x_2d = x_permuted.reshape(batch_size, feature_size)
    
    # Prepare output tensor
    out = torch.empty(batch_size, dtype=torch.long, device=x.device)
    
    # Set kernel parameters
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    # Launch kernel
    argmax_kernel[grid](x_2d, out, batch_size, feature_size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape output back to expected shape
    out_shape = [s for i, s in enumerate(shape) if i != dim]
    out = out.reshape(out_shape)
    
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