import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_normalize_kernel(
    x_ptr,  # Input tensor pointer
    out_ptr,  # Output tensor pointer
    n_rows,  # Number of rows (batch size)
    n_cols,  # Number of columns (dimension)
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate row offset
    row_start = row_idx * n_cols
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator for L1 norm
    l1_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process the row in blocks
    for start in range(0, n_cols, BLOCK_SIZE):
        # Calculate actual offsets for this block
        offsets = row_start + start + col_offsets
        # Create mask for valid elements
        mask = offsets < (row_start + n_cols)
        
        # Load input values
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute absolute value and accumulate
        abs_x = tl.abs(x)
        l1_sum += abs_x * tl.cast(mask, tl.float32)
    
    # Reduce within the block to get sum for this row
    # Since each program handles one row, we need to sum across the block dimension
    # For simplicity, we'll use a sequential reduction since the block size is fixed
    # In practice, we'd use tl.sum but for this simple case, we can just sum manually
    # Actually, tl.sum is available and efficient
    l1_sum = tl.sum(l1_sum, axis=0)
    
    # Compute mean: l1_mean = l1_sum / n_cols
    # But note: the original code uses torch.mean which divides by number of elements in the dimension
    # Since we're computing along dim=1, it's divided by n_cols
    l1_mean = l1_sum / tl.cast(n_cols, tl.float32)
    
    # Now perform the normalization: x / l1_mean
    # Process the row again for the division
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + start + col_offsets
        mask = offsets < (row_start + n_cols)
        
        # Load input values
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize
        out = x / l1_mean
        
        # Store result
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_l1_normalize(x: torch.Tensor):
    """
    Applies L1 normalization: x / mean(abs(x), dim=1, keepdim=True)
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    n_rows = x.size(0)
    n_cols = x.size(1)
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Determine block size (power of 2 for efficiency)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the Triton kernel
    l1_normalize_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using Triton kernel.
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
        return triton_l1_normalize(x)