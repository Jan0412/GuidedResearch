import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cumulative_sum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the block ID
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate which row we're processing
    row_id = tl.program_id(1)
    
    # Mask to avoid out-of-bounds access
    mask = offsets < dim_size
    
    # Load input data for this row
    x_row = tl.load(x_ptr + row_id * stride_x + offsets, mask=mask, other=0.0)
    
    # Compute cumulative sum
    cumsum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(dim_size):
        if i < BLOCK_SIZE:
            cumsum[i] = tl.where(i == 0, x_row[i], cumsum[i-1] + x_row[i])
    
    # Store results
    tl.store(out_ptr + row_id * stride_out + offsets, cumsum, mask=mask)

def triton_cumsum(x: torch.Tensor, dim: int):
    """
    Triton implementation of cumulative sum along a specific dimension
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get tensor properties
    shape = x.shape
    dim_size = shape[dim]
    batch_size = 1
    for i in range(len(shape)):
        if i != dim:
            batch_size *= shape[i]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine strides
    stride_x = x.stride()[dim] if len(x.stride()) > 0 else 1
    stride_out = out.stride()[dim] if len(out.stride()) > 0 else 1
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = (
        (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE,
        batch_size
    )
    
    # Launch kernel
    cumulative_sum_kernel[grid](
        x, out, 
        x.numel(), 
        dim_size,
        stride_x,
        stride_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum(x, self.dim)