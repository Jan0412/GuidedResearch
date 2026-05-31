import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    batch_size,  # Number of rows/batches
    dim,  # Dimension of each row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch dimension)
    batch_id = tl.program_id(0)
    
    # Compute row offset
    row_start = batch_id * dim
    
    # Create a range of offsets [0..dim-1]
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    
    # Load input values
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (online max)
    row_max = tl.max(x, axis=0)
    
    # Subtract max for numerical stability
    x_shifted = x - row_max
    
    # Compute exponentials
    x_exp = tl.exp(x_shifted)
    
    # Compute sum of exponentials
    row_sum = tl.sum(x_exp, axis=0)
    
    # Compute softmax: exp(x - max) / sum(exp(x - max))
    softmax_output = x_exp / row_sum
    
    # Store result
    tl.store(out_ptr + row_start + offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton implementation of softmax.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    batch_size = x.shape[0]
    dim = x.shape[1]
    
    # Use a reasonable block size (power of 2)
    BLOCK_SIZE = 1024  # Adjust based on typical dimension sizes
    
    # Grid: one block per batch
    grid = (batch_size,)
    
    # Launch the kernel
    softmax_kernel[grid](x, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Triton-optimized Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x, dim=1)