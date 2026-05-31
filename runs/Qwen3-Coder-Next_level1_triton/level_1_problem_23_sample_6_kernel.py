import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Number of rows
    dim,  # Dimension of each row
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for softmax along dimension 1.
    Uses online softmax for numerical stability.
    """
    # Each program handles one row (batch element)
    batch_id = tl.program_id(0)
    
    # Pointer to the start of this row
    row_start = batch_id * dim
    
    # Initialize max and sum for online softmax
    row_max = tl.full([1], -float("inf"), dtype=tl.float32)
    row_sum = tl.full([1], 0.0, dtype=tl.float32)
    
    # First pass: compute max and sum
    for offset in range(0, dim, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        
        # Update max
        row_max = tl.maximum(row_max, tl.max(x, axis=0, keep_dims=True))
        
    # Second pass: compute sum with max subtraction for stability
    for offset in range(0, dim, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        x = x.to(tl.float32)
        
        # Compute exponentials with max subtraction
        x_exp = tl.exp(x - row_max)
        
        # Update sum
        row_sum = row_sum + tl.sum(x_exp, axis=0)
        
    # Third pass: compute final softmax values
    for offset in range(0, dim, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        x = x.to(tl.float32)
        
        # Compute softmax
        softmax_val = tl.exp(x - row_max) / row_sum
        
        # Store result
        tl.store(out_ptr + row_start + offsets, softmax_val.to(x_ptr.dtype.element_ty), mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton-based softmax implementation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    batch_size = x.size(0)
    dim = x.size(1)
    
    # Determine block size (tuned for the given dimension)
    BLOCK_SIZE = 256
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    softmax_kernel[grid](x, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses Triton kernel for softmax.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-based Softmax activation to the input tensor.
        """
        return triton_softmax(x, dim=1)