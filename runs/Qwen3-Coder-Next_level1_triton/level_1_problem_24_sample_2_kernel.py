import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,  # Input pointer
    output_ptr,  # Output pointer
    batch_size,  # Number of rows
    dim,  # Dimension size
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch dimension)
    row_idx = tl.program_id(0)
    
    # Compute row offset
    row_start = row_idx * dim
    
    # Pointer to current row
    x_row_ptr = x_ptr + row_start
    output_row_ptr = output_ptr + row_start
    
    # Initialize max and sum for numerical stability
    row_max = -float("inf")
    row_sum = 0.0
    
    # First pass: compute max and sum of exponentials
    for col_start in range(0, dim, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < dim
        
        # Load data
        x_vals = tl.load(x_row_ptr + col_offsets, mask=mask, other=-float("inf"))
        
        # Compute max
        row_max = tl.maximum(row_max, tl.max(x_vals))
        
        # Compute sum of exponentials (offset by max for stability)
        exp_vals = tl.exp(x_vals - row_max)
        row_sum += tl.sum(exp_vals, mask=mask)
    
    # Compute log of sum
    log_sum = tl.log(row_sum) + row_max
    
    # Second pass: compute final log_softmax values
    for col_start in range(0, dim, BLOCK_SIZE):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < dim
        
        # Load data
        x_vals = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
        
        # Compute log_softmax: x - log_sum
        result = x_vals - log_sum
        
        # Store result
        tl.store(output_row_ptr + col_offsets, result, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of log_softmax.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply log_softmax
        
    Returns:
        Result of log_softmax operation
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    batch_size = x.size(0) if dim == 1 else x.size(1)
    dim_size = x.size(dim)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Determine grid size (one block per row)
    grid = (batch_size,)
    
    # Launch kernel
    log_softmax_kernel[grid](
        x, out, batch_size, dim_size, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation using Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return triton_log_softmax(x, dim=self.dim)