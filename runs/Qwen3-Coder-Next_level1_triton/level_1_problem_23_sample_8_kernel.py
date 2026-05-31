import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    batch_size,  # Number of rows
    dim,  # Dimension of each row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row (batch element)
    batch_idx = tl.program_id(0)
    
    # Calculate base pointers for this batch
    x_row_start = x_ptr + batch_idx * dim
    out_row_start = out_ptr + batch_idx * dim
    
    # Initialize max and sum for online softmax
    row_max = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # First pass: compute max and sum for numerical stability
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load input values
        x = tl.load(x_row_start + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        
        # Update max
        row_max = tl.maximum(row_max, x)
        
        # Update sum of exponentials (we'll compute this in second pass)
    
    # Reduce to get the actual max for this row
    row_max = tl.max(row_max, axis=0)
    
    # Second pass: compute exponentials and sum
    row_sum = tl.zeros([1], dtype=tl.float32)
    
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load input values
        x = tl.load(x_row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute exp(x - max) for numerical stability
        exp_x = tl.exp(x - row_max)
        
        # Accumulate sum
        row_sum += tl.sum(exp_x, axis=0)
        
        # Store intermediate exp values for third pass
        tl.store(out_row_start + offsets, exp_x, mask=mask)
    
    # Third pass: normalize by sum
    inv_sum = 1.0 / row_sum[0]
    
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load stored exp values
        exp_x = tl.load(out_row_start + offsets, mask=mask).to(tl.float32)
        
        # Normalize
        softmax_val = exp_x * inv_sum
        
        # Store result
        tl.store(out_row_start + offsets, softmax_val, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    Triton implementation of softmax along dimension 1.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
        
    Returns:
        Softmax applied tensor of same shape
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Tunable parameters
    BLOCK_SIZE = 512  # Good balance for large dimensions
    
    # Grid: one program per batch element
    grid = (batch_size,)
    
    # Launch kernel
    softmax_kernel[grid](
        x, out, batch_size, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model with Triton-based softmax.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-optimized Softmax activation to the input tensor.
        """
        return triton_softmax(x)