import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def logsoftmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    
    # Check if we're within bounds
    if row_idx >= n_rows:
        return
        
    # Calculate starting offset for this row
    row_start = row_idx * n_cols
    
    # Load the entire row into shared memory
    # Use a loop to handle large rows efficiently
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load input values for this chunk
        x_vals = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        
        # Compute max value for numerical stability
        max_val = tl.max(x_vals)
        
        # Compute exp(x - max_val) and sum
        exp_vals = tl.exp(x_vals - max_val)
        sum_exp = tl.sum(exp_vals)
        
        # Compute log(sum_exp) and store final result
        log_sum_exp = tl.log(sum_exp)
        final_result = x_vals - max_val - log_sum_exp
        
        # Store results back
        tl.store(out_ptr + row_start + offsets, final_result, mask=mask)

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
        # Ensure input is contiguous and on GPU
        x = x.contiguous().cuda()
        
        # Prepare output tensor
        out = torch.empty_like(x)
        
        # Get dimensions
        batch_size = x.shape[0]
        n_cols = x.shape[self.dim] if self.dim == 1 else x.shape[-1]
        
        # Define block size for Triton kernel
        BLOCK_SIZE = 1024  # This can be tuned based on hardware
        
        # Determine grid size (number of blocks needed)
        grid_size = (batch_size,)
        
        # Launch the Triton kernel
        logsoftmax_kernel[grid_size](
            x,
            out,
            batch_size,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out