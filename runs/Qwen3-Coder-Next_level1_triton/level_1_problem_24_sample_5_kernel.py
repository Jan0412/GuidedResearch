import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (dim)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input
    row_idx = tl.program_id(0)
    
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    
    # Create column offsets [0..BLOCK_SIZE-1]
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load the row data
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(x, axis=0)
    
    # Compute exp(x - max)
    x_minus_max = x - row_max
    exp_x = tl.exp(x_minus_max)
    
    # Compute sum of exp(x - max)
    exp_sum = tl.sum(exp_x, axis=0)
    
    # Compute log(sum(exp(x - max))) using log-sum-exp trick
    # log_sum = log(exp_sum) but we need to handle this carefully
    # Actually, we can compute: log_sum = row_max + tl.log(exp_sum)
    # But better to use: log_sum = row_max + tl.log(tl.sum(tl.exp(x - row_max)))
    
    # Compute log-sum-exp: log(sum(exp(x)))
    log_sum_exp = row_max + tl.log(exp_sum)
    
    # Compute log_softmax: x - log_sum_exp
    result = x - log_sum_exp
    
    # Store result
    tl.store(out_ptr + row_start + offsets, result, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    """
    Compute log_softmax using Triton kernel.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute log_softmax
        
    Returns:
        Output tensor with log_softmax applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    n_rows = 1
    for i in range(len(shape)):
        if i != dim:
            n_rows *= shape[i]
    n_cols = shape[dim]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Cap at reasonable maximum
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # Create grid - one block per row
    grid = (n_rows,)
    
    # Launch kernel
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using Triton kernel.
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