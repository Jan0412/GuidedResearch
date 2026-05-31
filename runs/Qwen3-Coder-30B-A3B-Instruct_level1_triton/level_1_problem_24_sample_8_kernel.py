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
    
    # Calculate the start position for this row
    input_row_start = row_idx * n_cols
    output_row_start = row_idx * n_cols
    
    # Create a range of column indices for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process the row in chunks of BLOCK_SIZE
    for col_start in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets
        offsets = col_start + col_offsets
        
        # Create mask to avoid going out of bounds
        mask = offsets < n_cols
        
        # Load input values
        input_vals = tl.load(input_ptr + input_row_start + offsets, mask=mask, other=-float('inf'))
        
        # Compute max value for numerical stability
        max_val = tl.max(input_vals)
        
        # Subtract max from all values to prevent overflow
        shifted_vals = input_vals - max_val
        
        # Compute exp values
        exp_vals = tl.exp(shifted_vals)
        
        # Compute sum of exponentials
        sum_exp = tl.sum(exp_vals)
        
        # Compute log(sum_exp) + max_val (which is actually just log(sum_exp) due to the subtraction)
        log_sum_exp = tl.log(sum_exp)
        
        # Compute final log_softmax value
        log_softmax_vals = shifted_vals - log_sum_exp
        
        # Store results
        tl.store(output_ptr + output_row_start + offsets, log_softmax_vals, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for LogSoftmax activation.
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
        n_cols = x.shape[self.dim]
        
        # Define block size (can be tuned)
        BLOCK_SIZE = 1024
        
        # Calculate grid size
        grid = (batch_size,)
        
        # Launch kernel
        logsoftmax_kernel[grid](
            x,
            output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output