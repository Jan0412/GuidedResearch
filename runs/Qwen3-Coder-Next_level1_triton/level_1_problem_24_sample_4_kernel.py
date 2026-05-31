import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    X,  # pointer to input tensor
    Y,  # pointer to output tensor
    N,  # number of columns (dimension)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input
    row_start = tl.program_id(0)
    row_offset = row_start * N
    
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load the row data
    x = tl.load(X + row_offset + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    x_max = tl.max(x, axis=0)
    
    # Compute exponentials of (x - max)
    x_exp = tl.exp(x - x_max)
    
    # Compute sum of exponentials
    x_sum = tl.sum(x_exp, axis=0)
    
    # Compute log sum exp: log(sum(exp(x - max))) + max
    log_sum_exp = tl.log(x_sum) + x_max
    
    # Compute log softmax: x - log_sum_exp
    result = x - log_sum_exp
    
    # Store the result
    tl.store(Y + row_offset + offsets, result, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = -1):
    """
    Triton implementation of log_softmax.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    assert dim == 1, "This implementation only supports dim=1 for simplicity."
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Prepare output tensor
    y = torch.empty_like(x)
    
    # Get dimensions
    batch_size, N = x.shape
    
    # Determine block size (power of 2 that's >= N, but not too large)
    BLOCK_SIZE = 1
    while BLOCK_SIZE < N:
        BLOCK_SIZE *= 2
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)  # Cap at 8192 for practicality
    
    # Define grid: one block per row
    grid = (batch_size,)
    
    # Launch the kernel
    log_softmax_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return y


class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation using custom Triton kernel.
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