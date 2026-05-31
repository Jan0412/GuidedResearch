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
    
    # Calculate the starting position for this row
    input_row_start = row_idx * n_cols
    output_row_start = row_idx * n_cols
    
    # Load input values for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load input values with masking
    input_vals = tl.load(input_ptr + input_row_start + offsets, mask=mask, other=-float('inf'))
    
    # Find max value in the row for numerical stability
    max_val = tl.max(input_vals)
    
    # Compute exp(x - max_val) for each element
    exp_vals = tl.exp(input_vals - max_val)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)
    
    # Compute log_softmax: x - max - log(sum_exp)
    log_softmax_vals = input_vals - max_val - log_sum_exp
    
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
        n_cols = x.shape[self.dim] if self.dim == 1 else x.shape[-1]
        
        # Define block size
        BLOCK_SIZE = 1024  # Tuned for optimal performance
        
        # Grid size
        grid = (batch_size,)
        
        # Launch kernel
        logsoftmax_kernel[grid](
            x,
            output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output