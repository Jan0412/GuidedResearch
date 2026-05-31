import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (features per row)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input
    row_id = tl.program_id(0)
    
    # Calculate starting offsets for this row
    row_start = row_id * n_cols
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max value and sum for softmax
    row_max = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    row_sum = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
    
    # Process the row in chunks to find max and compute sum of exponentials
    for start in range(0, n_cols, BLOCK_SIZE):
        # Create mask for valid elements
        mask = (start + col_offsets) < n_cols
        # Load data as float32 for numerical stability
        x = tl.load(x_ptr + row_start + start + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        
        # Update max value using maximum operation
        row_max = tl.maximum(row_max, x)
        
        # Compute exponentials relative to current max for numerical stability
        x_exp = tl.exp(x - row_max)
        row_sum += x_exp
    
    # Final max reduction across the block
    row_max = tl.max(row_max, axis=0)
    
    # Recompute sum with final max value
    row_sum = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        mask = (start + col_offsets) < n_cols
        x = tl.load(x_ptr + row_start + start + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
        x_exp = tl.exp(x - row_max)
        row_sum += tl.sum(x_exp)
    
    # Compute final softmax output
    for start in range(0, n_cols, BLOCK_SIZE):
        mask = (start + col_offsets) < n_cols
        x = tl.load(x_ptr + row_start + start + col_offsets, mask=mask, other=0.0).to(tl.float32)
        softmax_out = tl.exp(x - row_max) / row_sum
        tl.store(out_ptr + row_start + start + col_offsets, softmax_out.to(x_ptr.dtype.element_ty), mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Triton-based softmax implementation.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax
        
    Returns:
        Output tensor with softmax applied
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    n_rows = x.size(0) if dim == 1 else x.size(1)
    n_cols = x.size(1) if dim == 1 else x.size(0)
    
    # Set block size
    BLOCK_SIZE = 256
    
    # Determine grid size
    grid = lambda meta: (n_rows,)
    
    # Launch kernel
    softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses Triton kernel for softmax operation.
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
        return triton_softmax(x, dim=1)