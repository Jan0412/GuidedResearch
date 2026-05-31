import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    X_ptr,  # Pointer to input
    Y_ptr,  # Pointer to output
    batch_size,  # Number of rows
    dim,  # Number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row (one softmax computation)
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * dim
    
    # Offset pointers to the start of this row
    x_row_ptr = X_ptr + row_start
    y_row_ptr = Y_ptr + row_start
    
    # Initialize max and sum for online softmax algorithm
    row_max = tl.full([1], -float("inf"), dtype=tl.float32)
    row_sum = tl.full([1], 0.0, dtype=tl.float32)
    
    # First pass: compute max and sum for numerical stability
    for col_offset in range(0, dim, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < dim
        
        # Load input values
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        
        # Update max
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
        
    # Second pass: compute exp(x - max) and sum
    for col_offset in range(0, dim, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < dim
        
        # Load input values
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        
        # Compute exp(x - max) for numerical stability
        exp_x = tl.exp(x - row_max)
        
        # Update sum
        row_sum = row_sum + tl.sum(exp_x, axis=0)
        
        # Store exponentiated values temporarily (we'll normalize later)
        tl.store(y_row_ptr + col_offsets, exp_x, mask=mask)
    
    # Third pass: normalize by dividing by sum
    inv_sum = 1.0 / row_sum
    for col_offset in range(0, dim, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < dim
        
        # Load the exponentiated values
        exp_x = tl.load(y_row_ptr + col_offsets, mask=mask)
        
        # Normalize and store final result
        softmax_val = exp_x * inv_sum
        tl.store(y_row_ptr + col_offsets, softmax_val.to(tl.float32), mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton-based softmax implementation using online softmax algorithm.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    batch_size, dim_size = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 256
    
    # Grid: one program per row
    grid = (batch_size,)
    
    # Launch the kernel
    softmax_kernel[grid](
        x, out, batch_size, dim_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using Triton kernel.
        """
        return triton_softmax(x, dim=1)