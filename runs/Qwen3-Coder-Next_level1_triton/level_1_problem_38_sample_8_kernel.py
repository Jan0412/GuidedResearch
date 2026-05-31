import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows
    n_cols,  # Number of columns per row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Accumulator for L1 norm (sum of absolute values)
    l1_sum = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over columns in blocks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Compute absolute value and accumulate
        abs_x = tl.abs(x)
        l1_sum += tl.sum(abs_x, axis=0)
    
    # Compute mean (l1_sum / n_cols)
    mean_val = l1_sum / n_cols
    
    # Second pass: normalize the row
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Normalize
        out = x / mean_val
        
        # Store result
        tl.store(out_ptr + row_start + col_offsets, out, mask=mask)


def triton_l1_norm(x: torch.Tensor):
    """
    Apply L1 normalization using Triton kernel.
    Normalizes by dividing by the mean of absolute values along dim=1.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    if len(shape) < 2:
        # Handle case where dim=1 doesn't exist by adding a dimension
        x = x.unsqueeze(1)
        shape = x.shape
    
    n_rows = shape[0]
    n_cols = shape[1]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size
    BLOCK_SIZE = 512  # Tunable parameter
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    l1_norm_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using Triton kernels.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return triton_l1_norm(x)