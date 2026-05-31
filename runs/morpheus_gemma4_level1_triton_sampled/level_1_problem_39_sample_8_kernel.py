import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def norm_kernel(
    x_ptr, 
    norm_ptr, 
    dim, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Create offsets for the block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Accumulator for sum of squares
    acc = 0.0
    
    # Loop over the dimension in blocks
    for i in range(0, dim, BLOCK_SIZE):
        # Mask for boundary conditions
        mask = (i + offsets) < dim
        # Load elements of the row
        val = tl.load(x_ptr + row_idx * dim + i + offsets, mask=mask, other=0.0)
        # Accumulate sum of squares
        acc += tl.sum(val * val, axis=0)
    
    # Compute L2 norm and store it
    tl.store(norm_ptr + row_idx, tl.sqrt(acc))

@triton.jit
def div_kernel(
    x_ptr, 
    norm_ptr, 
    out_ptr, 
    dim, 
    BLOCK_SIZE: tl.constexpr
):
    # Program ID for row and block of columns
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    
    # Create offsets for the current column block
    offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    
    # Load the precomputed norm for this row
    norm = tl.load(norm_ptr + row_idx)
    
    # Load input values
    val = tl.load(x_ptr + row_idx * dim + offsets, mask=mask, other=0.0)
    
    # Perform division and store the result
    tl.store(out_ptr + row_idx * dim + offsets, val / norm, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    """
    Triton implementation of L2 normalization along dim=1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # Output tensor
    out = torch.empty_like(x)
    # Intermediate tensor to store norms for each row
    norms = torch.empty((batch_size,), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 1024
    
    # 1. Compute norms
    norm_grid = (batch_size,)
    norm_kernel[norm_grid](
        x, norms, dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 2. Divide elements by norms
    div_grid = (batch_size, (dim + BLOCK_SIZE - 1) // BLOCK_SIZE)
    div_kernel[div_grid](
        x, norms, out, dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        return triton_l2_norm(x)