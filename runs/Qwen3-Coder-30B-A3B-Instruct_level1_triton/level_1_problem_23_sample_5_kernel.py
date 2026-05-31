import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    inp_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting position for this row
    inp_row_start = row_idx * n_cols
    out_row_start = row_idx * n_cols
    
    # Load the row data
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Process in chunks of BLOCK_SIZE
    for i in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets for this chunk
        chunk_offsets = i + offsets
        chunk_mask = chunk_offsets < n_cols
        
        # Load input values
        inp_vals = tl.load(inp_ptr + inp_row_start + chunk_offsets, mask=chunk_mask, other=-float('inf'))
        
        # Compute max value for numerical stability
        max_val = tl.max(inp_vals)
        
        # Subtract max for numerical stability
        shifted_vals = inp_vals - max_val
        
        # Compute exp
        exp_vals = tl.exp(shifted_vals)
        
        # Compute sum of exponentials
        sum_exp = tl.sum(exp_vals)
        
        # Compute softmax
        softmax_vals = exp_vals / sum_exp
        
        # Store results
        tl.store(out_ptr + out_row_start + chunk_offsets, softmax_vals, mask=chunk_mask)

class ModelNew(nn.Module):
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
        batch_size, num_features = x.shape
        
        # Ensure input is contiguous
        x = x.contiguous().to(torch.float32)
        
        # Create output tensor
        output = torch.empty_like(x)
        
        # Define block size
        BLOCK_SIZE = 1024
        
        # Calculate grid size
        grid = (batch_size,)
        
        # Launch kernel
        softmax_kernel[grid](
            x,
            output,
            num_features,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output