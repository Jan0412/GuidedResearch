import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (dim)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of the input
    row_idx = tl.program_id(0)
    
    # Calculate row offset
    row_start = row_idx * n_cols
    
    # Create a range of column offsets [0..n_cols-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load the row data
    row = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (online max reduction)
    row_max = tl.max(row, axis=0)
    
    # Subtract max and exponentiate
    row_shifted = row - row_max
    row_exp = tl.exp(row_shifted)
    
    # Compute sum of exponentials (online sum reduction)
    row_sum = tl.sum(row_exp, axis=0)
    
    # Compute log(sum(exp(x))) using log-sum-exp trick
    log_sum_exp = row_max + tl.log(row_sum)
    
    # Compute log_softmax: x - log_sum_exp
    result = row - log_sum_exp
    
    # Store the result
    tl.store(out_ptr + row_start + col_offsets, result, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    """
    Computes log_softmax using a custom Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to compute log_softmax
        
    Returns:
        torch.Tensor: Output tensor with log_softmax applied
    """
    # Ensure tensor is contiguous and on GPU
    x = x.contiguous()
    assert x.is_cuda, "Tensor must be on CUDA device"
    
    # Get shape info
    shape = x.shape
    n_rows = 1
    n_cols = shape[dim]
    
    # Calculate number of rows by flattening all dimensions except dim
    for i, s in enumerate(shape):
        if i != dim:
            n_rows *= s
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = 2048  # Good default for large dimensions like 393216
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the kernel
    log_softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
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
        Applies LogSoftmax activation using Triton kernel to the input tensor.
        """
        return triton_log_softmax(x, dim=self.dim)