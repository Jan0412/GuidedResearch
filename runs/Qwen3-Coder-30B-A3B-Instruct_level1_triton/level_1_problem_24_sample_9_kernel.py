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
    input_row_ptr = input_ptr + row_idx * n_cols
    output_row_ptr = output_ptr + row_idx * n_cols
    
    # Create offsets for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process the row in chunks
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets for this chunk
        chunk_offsets = i + offsets
        
        # Create mask to avoid going out of bounds
        mask = chunk_offsets < n_cols
        
        # Load input values
        input_vals = tl.load(input_row_ptr + chunk_offsets, mask=mask, other=-float('inf'))
        
        # Compute max value for numerical stability
        max_val = tl.max(input_vals)
        
        # Subtract max and compute exp
        exp_vals = tl.exp(input_vals - max_val)
        
        # Compute sum of exponentials
        sum_exp = tl.sum(exp_vals)
        
        # Compute log(sum_exp) + max_val
        log_sum_exp = tl.log(sum_exp) + max_val
        
        # Compute log_softmax = input - log_sum_exp
        log_softmax_vals = input_vals - log_sum_exp
        
        # Store results
        tl.store(output_row_ptr + chunk_offsets, log_softmax_vals, mask=mask)

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
        batch_size, n_cols = x.shape
        
        # Define block size
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