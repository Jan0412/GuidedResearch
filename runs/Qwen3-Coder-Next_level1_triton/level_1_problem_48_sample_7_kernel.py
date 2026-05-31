import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_reduce_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_cols,  # Number of elements in the reduction dimension
    stride_row,  # Stride between consecutive rows in input
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (all elements in that row across the reduction dimension)
    row_idx = tl.program_id(0)
    
    # Calculate starting pointer for this row
    x_row_start = x_ptr + row_idx * stride_row
    
    # Initialize sum accumulator
    sum = tl.zeros([1], dtype=tl.float32)
    
    # Process in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        # Compute offsets for this block
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data (with masking)
        x = tl.load(x_row_start + offsets, mask=mask, other=0.0)
        
        # Accumulate sum
        sum = sum + tl.sum(x, axis=0)
    
    # Compute mean
    mean = sum / n_cols
    
    # Store result
    tl.store(out_ptr + row_idx, mean)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes mean reduction along specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Tensor with the specified dimension reduced by taking the mean
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Handle negative dimension indexing
    if dim < 0:
        dim = x.ndim + dim
    
    # Get shape information
    shape = x.shape
    n_rows = 1
    for i in range(dim):
        n_rows *= shape[i]
    
    n_cols = shape[dim]
    
    # Compute output shape
    out_shape = list(shape)
    del out_shape[dim]
    out_shape = tuple(out_shape)
    
    # Prepare output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Calculate stride for the reduction dimension
    stride = x.stride(dim)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 256
    
    # Determine grid size (one block per row)
    grid = (n_rows,)
    
    # Launch kernel
    mean_reduce_kernel[grid](
        x, out, n_cols, stride, BLOCK_SIZE=BLOCK_SIZE
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
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        return triton_mean(x, self.dim)