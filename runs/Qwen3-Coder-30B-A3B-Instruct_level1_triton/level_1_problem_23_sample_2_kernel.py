import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
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
    
    # Load data for this row
    offsets = input_row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_idx + 1) * n_cols
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Subtract max for numerical stability
    row_max = tl.max(input_vals, axis=0)
    input_vals = input_vals - row_max
    
    # Compute exp
    exp_vals = tl.exp(input_vals)
    
    # Compute sum of exps
    row_sum = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    softmax_vals = exp_vals / row_sum
    
    # Store result
    tl.store(output_ptr + offsets, softmax_vals, mask=mask)

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for Softmax activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Ensure input is contiguous and on GPU
        x = x.contiguous().cuda()
        
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Get dimensions
        batch_size, n_cols = x.shape
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid size
        grid = (batch_size,)
        
        # Launch kernel
        softmax_kernel[grid](
            x,
            output,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output