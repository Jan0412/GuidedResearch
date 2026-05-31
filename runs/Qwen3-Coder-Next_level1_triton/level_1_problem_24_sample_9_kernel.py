import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    X,  # Pointer to input
    Y,  # Pointer to output
    stride_x,  # Stride for input tensor
    stride_y,  # Stride for output tensor
    N: tl.constexpr,  # Number of elements per row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load the row data
    mask = col_offsets < N
    x_row = tl.load(X + row_start * stride_x + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    x_max = tl.max(x_row, axis=0)
    
    # Compute exp(x - max)
    x_shifted = x_row - x_max
    exp_x = tl.exp(x_shifted)
    
    # Compute sum of exp(x - max)
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Compute log(sum(exp(x - max))) + max (log-sum-exp trick)
    log_sum_exp = tl.log(sum_exp) + x_max
    
    # Compute log_softmax = x - log(sum(exp(x)))
    log_softmax = x_row - log_sum_exp
    
    # Store result
    tl.store(Y + row_start * stride_y + col_offsets, log_softmax, mask=mask)


def triton_log_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton-based log_softmax implementation optimized for large dimensions.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
        dim: Dimension along which to apply log_softmax
        
    Returns:
        Output tensor with log_softmax applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    assert dim == 1, "Only dim=1 is supported for this implementation"
    
    batch_size, dim_size = shape[0], shape[1]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size - tuned for large dimensions
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = (batch_size,)
    
    # Calculate strides
    stride_x = x.stride(0)
    stride_y = out.stride(0)
    
    # Launch kernel
    log_softmax_kernel[grid](
        X=x,
        Y=out,
        stride_x=stride_x,
        stride_y=stride_y,
        N=dim_size,
        BLOCK_SIZE=BLOCK_SIZE,
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
        Applies LogSoftmax activation to the input tensor using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        return triton_log_softmax(x, dim=self.dim)