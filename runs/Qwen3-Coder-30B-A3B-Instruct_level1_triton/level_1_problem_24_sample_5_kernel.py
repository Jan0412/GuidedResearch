import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def logsoftmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting column index for this thread block
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # Create a range of column indices
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid columns
    mask = col_offsets < n_cols
    
    # Load input data
    input_vals = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max value for numerical stability
    max_val = tl.max(input_vals, axis=0)
    
    # Subtract max from all values to prevent overflow
    shifted_vals = input_vals - max_val
    
    # Compute exp and sum
    exp_vals = tl.exp(shifted_vals)
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute log(sum_exp) + max_val
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Compute final log_softmax value
    output_vals = shifted_vals - log_sum_exp
    
    # Store result
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output_vals, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for LogSoftmax activation.
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
        # Ensure input is contiguous and on GPU
        x = x.contiguous().cuda()
        
        # Create output tensor
        output = torch.empty_like(x)
        
        # Get dimensions
        batch_size, n_cols = x.shape
        
        # Configure block size
        BLOCK_SIZE = 1024
        
        # Calculate grid dimensions
        grid = (
            batch_size,  # One block per row
            (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE  # Number of blocks per row
        )
        
        # Launch kernel
        logsoftmax_kernel[grid](
            x,
            output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output