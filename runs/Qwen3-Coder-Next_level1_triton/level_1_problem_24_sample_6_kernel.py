import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element)
    batch_id = tl.program_id(0)
    
    # Calculate base pointers for this batch
    x_offset = batch_id * dim
    out_offset = batch_id * dim
    
    # Initialize max and sum for log-softmax
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    sum_val = tl.full([1], 0.0, dtype=tl.float32)
    
    # First pass: find max and compute sum of exp(x - max)
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load data
        x = tl.load(x_ptr + x_offset + offsets, mask=mask, other=-float('inf'))
        x = x.to(tl.float32)
        
        # Update max
        row_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, row_max)
        
        # Update sum with shifted exponentials
        exp_x = tl.exp(x - max_val)
        sum_val = sum_val + tl.sum(exp_x, axis=0)
    
    # Compute log-sum-exp
    log_sum = tl.log(sum_val) + max_val
    
    # Second pass: compute log_softmax = x - log_sum
    for start in range(0, dim, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load data
        x = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0)
        x = x.to(tl.float32)
        
        # Compute log_softmax
        log_softmax_val = x - log_sum
        
        # Store result
        tl.store(out_ptr + out_offset + offsets, log_softmax_val.to(x_ptr.dtype.element_ty), mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of log_softmax.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size = x.size(0) if dim == 1 else x.size(0)
    actual_dim = x.size(dim)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    log_softmax_kernel[grid](
        x, out, batch_size, actual_dim,
        BLOCK_SIZE=BLOCK_SIZE,
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
        Applies LogSoftmax activation to the input tensor using Triton kernel.
        """
        return triton_log_softmax(x, dim=self.dim)