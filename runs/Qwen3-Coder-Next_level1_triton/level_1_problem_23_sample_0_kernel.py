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
    # Each program handles one row of the softmax computation
    row_start = tl.program_id(0)
    
    # Create column offsets for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load the row data
    row_mask = col_offsets < n_cols
    x_row = tl.load(x_ptr + row_start * n_cols + col_offsets, mask=row_mask, other=-float('inf'))
    
    # For numerical stability, subtract the max value (online softmax trick)
    row_max = tl.max(x_row, axis=0)
    
    # Compute exponentials with numerical stability
    x_exp = tl.exp(x_row - row_max)
    
    # Compute sum of exponentials
    x_sum = tl.sum(x_exp, axis=0)
    
    # Compute softmax: exp(x - max) / sum(exp(x - max))
    softmax_output = x_exp / x_sum
    
    # Store the result
    tl.store(output_ptr + row_start * n_cols + col_offsets, softmax_output, mask=row_mask)


def triton_softmax(x: torch.Tensor):
    """
    Triton implementation of softmax along dimension 1.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features)
    
    Returns:
        torch.Tensor: Softmax applied tensor of same shape as input
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Set block size (tunable parameter)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 131072:  # Cap at 128K for practicality
        BLOCK_SIZE = 131072
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the kernel
    softmax_kernel[grid](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Softmax activation using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
        
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)