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
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Calculate starting offsets for this row
    row_start = row_idx * n_cols
    
    # Accumulate L1 norm for this row
    l1_sum = 0.0
    # Process in blocks to handle large n_cols
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load elements
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        # Compute absolute values and accumulate
        l1_sum += tl.sum(tl.abs(x), mask=mask)
    
    # Compute mean (divide by n_cols)
    mean_val = l1_sum / n_cols
    
    # Normalize: x / mean_val
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load elements
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        # Normalize
        out = x / mean_val
        # Store result
        tl.store(out_ptr + row_start + offsets, out, mask=mask)


def triton_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L1 normalization: x / mean(abs(x), dim=1, keepdim=True)
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        
    Returns:
        torch.Tensor: L1 normalized tensor with same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    assert x.dim() == 2, "Input must be 2D tensor"
    
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024  # Tunable parameter
    
    # Launch one block per row
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