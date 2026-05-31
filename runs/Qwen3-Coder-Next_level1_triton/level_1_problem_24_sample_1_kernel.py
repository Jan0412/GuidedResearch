import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    X,  # Pointer to input
    Y,  # Pointer to output
    batch_size,  # Number of batches
    dim,  # Dimension of each batch
    BLOCK_SIZE: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    
    # Calculate the starting offset for this batch
    batch_offset = batch_idx * dim
    
    # Create base pointers for this batch
    x_ptr = X + batch_offset
    y_ptr = Y + batch_offset
    
    # Initialize maximum and sum for log-sum-exp
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    sum_exp = tl.full([1], 0.0, dtype=tl.float32)
    
    # First pass: find maximum and compute sum of exponentials
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf')).to(tl.float32)
        
        # Update max
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
        
        # Update sum of exponentials (with numerical stability)
        exp_vals = tl.exp(x - max_val)
        sum_exp = sum_exp + tl.sum(exp_vals, axis=0)
    
    # Compute log(sum_exp) + max_val (the log-sum-exp)
    log_sum_exp = max_val + tl.log(sum_exp)
    
    # Second pass: compute log_softmax
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute log_softmax: x - log_sum_exp
        result = x - log_sum_exp
        
        # Store result
        tl.store(y_ptr + offsets, result.to(Y.dtype.element_ty), mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    """
    Apply log_softmax using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply log_softmax
        
    Returns:
        Output tensor with log_softmax applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert dim == 1, "This implementation assumes dim=1"
    
    # Ensure contiguous
    x = x.contiguous()
    
    # Prepare output
    out = torch.empty_like(x)
    
    batch_size = x.size(0)
    dim = x.size(1)
    
    # Set block size - large enough for good utilization but not too large
    BLOCK_SIZE = 1024
    
    # Grid: one block per batch
    grid = (batch_size,)
    
    # Launch kernel
    log_softmax_kernel[grid](
        x, out, batch_size, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using custom Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return triton_log_softmax(x, dim=self.dim)