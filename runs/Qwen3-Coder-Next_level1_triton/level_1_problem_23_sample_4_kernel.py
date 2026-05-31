import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row (batch element)
    batch_id = tl.program_id(0)
    
    # Calculate starting offset for this batch
    input_row_start = batch_id * dim
    output_row_start = batch_id * dim
    
    # Initialize max and sum for online softmax
    row_max = tl.full((1,), -float("inf"), dtype=tl.float32)
    row_sum = tl.full((1,), 0.0, dtype=tl.float32)
    
    # First pass: compute max and sum with numerical stability
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load input values
        x = tl.load(input_ptr + input_row_start + offsets, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        
        # Compute max reduction
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Second pass: compute sum of exponentials (relative to max for stability)
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load input values
        x = tl.load(input_ptr + input_row_start + offsets, mask=mask, other=0.0)
        x = x.to(tl.float32)
        
        # Compute exp(x - max) for numerical stability
        exp_x = tl.exp(x - row_max)
        
        # Compute sum
        row_sum = row_sum + tl.sum(exp_x, axis=0)
        
        # Store exp(x - max) temporarily for third pass
        tl.store(output_ptr + output_row_start + offsets, exp_x, mask=mask)
    
    # Third pass: normalize by sum
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        
        # Load the stored exponentials
        exp_x = tl.load(output_ptr + output_row_start + offsets, mask=mask, other=0.0)
        
        # Normalize and store result
        softmax_val = exp_x / row_sum
        tl.store(output_ptr + output_row_start + offsets, softmax_val, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    Triton-based softmax implementation.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
        
    Returns:
        Softmax applied tensor of same shape
    """
    assert x.is_cuda, "Input must be on CUDA device"
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Configure kernel parameters
    BLOCK_SIZE = 256  # Tunable parameter
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    softmax_kernel[grid](
        x,
        output,
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for softmax operation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softmax activation using Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
            
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)