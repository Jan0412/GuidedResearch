import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,  # Input pointer
    output_ptr,  # Output pointer
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (features)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input matrix
    row_idx = tl.program_id(0)
    
    # Start pointer for this row
    row_start = row_idx * n_cols
    
    # Load the row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load the row data
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability (online softmax trick)
    row_max = tl.max(x, axis=0)
    
    # Subtract max for numerical stability
    x_shifted = x - row_max
    
    # Compute exponentials
    exp_x = tl.exp(x_shifted)
    
    # Compute sum of exponentials
    exp_sum = tl.sum(exp_x, axis=0)
    
    # Compute log sum for normalization
    log_sum = tl.log(exp_sum)
    
    # Final softmax: exp(x - max) / sum(exp(x - max)) = exp(x - max - log_sum)
    # This is equivalent to exp(x) / sum(exp(x)) but more numerically stable
    softmax_result = tl.exp(x_shifted - log_sum)
    
    # Store the result
    tl.store(output_ptr + row_start + offsets, softmax_result, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1):
    """
    Triton-based softmax implementation optimized for FP32.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax
        
    Returns:
        Softmax applied tensor
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    if dim < 0:
        dim = len(shape) + dim
    
    # For 2D case (batch_size, num_features) with dim=1
    if len(shape) == 2 and dim == 1:
        n_rows = shape[0]
        n_cols = shape[1]
        
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Set block size - tune based on feature dimension
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        if BLOCK_SIZE > 131072:  # Cap at 128K
            BLOCK_SIZE = 131072
        
        # Grid: one program per row
        grid = (n_rows,)
        
        # Launch kernel
        softmax_kernel[grid](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        return output
    else:
        # Fallback to PyTorch for non-standard shapes
        return torch.softmax(x, dim=dim)


class ModelNew(nn.Module):
    """
    Optimized model with Triton-based softmax activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softmax activation to the input tensor using Triton kernel.
        """
        return triton_softmax(x, dim=1)