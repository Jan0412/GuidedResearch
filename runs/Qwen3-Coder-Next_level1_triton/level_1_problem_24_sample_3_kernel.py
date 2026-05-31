import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    X_ptr,  # Input pointer
    Y_ptr,  # Output pointer
    batch_size,  # Number of rows
    dim,  # Dimension of each row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row (one batch element)
    row_idx = tl.program_id(0)
    
    # Calculate row offset
    row_start = row_idx * dim
    
    # Create offsets for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    
    # Load the row data
    x = tl.load(X_ptr + row_start + offsets, mask=mask, other=-float('inf'))
    
    # Find max for numerical stability
    x_max = tl.max(x, axis=0)
    
    # Compute exp(x - max) with masking
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    
    # Compute sum of exp values
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Compute log(sum(exp)) using log-sum-exp trick
    log_sum_exp = x_max + tl.log(sum_exp)
    
    # Compute final log-softmax: x - log_sum_exp
    result = x - log_sum_exp
    
    # Store the result
    tl.store(Y_ptr + row_start + offsets, result, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Triton-based log-softmax implementation.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute log-softmax
        
    Returns:
        Tensor with log-softmax applied
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    dim_size = shape[dim]
    
    # Reshape to 2D for easier processing
    if dim != len(shape) - 1:
        # Move the target dimension to the last position
        perm = list(range(len(shape)))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(perm)
        shape_perm = x.shape
    
    # Flatten to 2D: [batch_size, dim]
    batch_size = 1
    for s in shape[:-1]:
        batch_size *= s
    x_flat = x.view(batch_size, dim_size)
    
    # Prepare output tensor
    y = torch.empty_like(x_flat)
    
    # Configure kernel parameters
    BLOCK_SIZE = triton.next_power_of_2(dim_size)
    # Ensure BLOCK_SIZE is at least 1
    BLOCK_SIZE = max(1, BLOCK_SIZE)
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Launch kernel
    log_softmax_kernel[grid](
        x_flat, y, batch_size, dim_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to original shape
    if dim != len(shape) - 1:
        # Reverse the permutation
        inv_perm = [0] * len(shape)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        y = y.view(shape_perm).permute(inv_perm)
    
    return y.view(shape)


class ModelNew(nn.Module):
    """
    Optimized model with Triton-based LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-optimized LogSoftmax activation to the input tensor.
        """
        return triton_log_softmax(x, dim=self.dim)