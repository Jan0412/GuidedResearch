import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_cols,  # Number of columns in the reduction dimension
    n_blocks,  # Number of blocks to process (batch_size * other_dims)
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
):
    # Each program handles one block (one output element)
    block_idx = tl.program_id(0)
    
    # Calculate the offset to the start of this block in the input
    x_block_start = block_idx * n_cols
    
    # Accumulate sum
    sum = tl.zeros([1], dtype=DTYPE)
    
    # Process in chunks of BLOCK_SIZE
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + x_block_start + cols, mask=mask, other=0.0)
        sum += tl.sum(x, axis=0)
    
    # Compute mean
    mean = sum / n_cols
    
    # Store result
    tl.store(out_ptr + block_idx, mean)


def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes mean reduction along specified dimension using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension to reduce over
        
    Returns:
        Output tensor with reduced dimension
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get original shape and compute output shape
    shape = list(x.shape)
    n_dims = len(shape)
    
    # Normalize dimension
    if dim < 0:
        dim += n_dims
    
    # Compute the size of the dimension we're reducing over
    dim_size = shape[dim]
    
    # Calculate total number of blocks (elements in output tensor)
    n_blocks = 1
    for i, s in enumerate(shape):
        if i != dim:
            n_blocks *= s
    
    # Prepare output tensor
    out_shape = shape[:dim] + shape[dim+1:]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set kernel parameters
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Launch kernel
    grid = (n_blocks,)
    mean_kernel[grid](
        x, out,
        dim_size, n_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=tl.float32,
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