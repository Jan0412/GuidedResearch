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
    
    # Calculate the starting column for this thread block
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    # Create offsets for the current row
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_cols
    
    # Load input values for this row
    input_row = tl.load(input_ptr + row_idx * n_cols + offsets, mask=mask, other=-float('inf'))
    
    # Find max value in the row for numerical stability
    max_val = tl.max(input_row, axis=0)
    
    # Subtract max for numerical stability
    shifted = input_row - max_val
    
    # Compute exp
    exp_val = tl.exp(shifted)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_val, axis=0)
    
    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)
    
    # Compute final log_softmax
    result = shifted - log_sum_exp
    
    # Store result
    tl.store(output_ptr + row_idx * n_cols + offsets, result, mask=mask)

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
        # Ensure input is on GPU and contiguous
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Get dimensions
        batch_size = x.shape[0]
        n_cols = x.shape[1]
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid dimensions
        grid = (
            batch_size,
            (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
        
        # Launch kernel
        logsoftmax_kernel[grid](
            x,
            output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output