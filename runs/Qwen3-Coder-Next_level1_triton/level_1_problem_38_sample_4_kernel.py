import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_normalize_kernel(
    x_ptr,  # Input tensor
    out_ptr,  # Output tensor
    n_rows,  # Number of rows
    n_cols,  # Number of columns (dimension)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for L1 normalization: x / mean(|x|, dim=1)
    which is equivalent to x / (sum(|x|, dim=1) / n_cols)
    = x * n_cols / sum(|x|, dim=1)
    """
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute the sum of absolute values for this row
    row_start_ptr = x_ptr + row_idx * n_cols
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Block loop to handle large rows
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        acc += tl.sum(tl.abs(x))
    
    # Compute mean = sum / n_cols
    mean = acc / n_cols
    
    # Normalize the row
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
        out = x / mean
        tl.store(out_ptr + row_idx * n_cols + col_offsets, out, mask=mask)


def triton_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L1 normalization to the input tensor using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        
    Returns:
        torch.Tensor: L1 normalized tensor with same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    assert x.dim() == 2, "Input must be 2D tensor"
    n_rows, n_cols = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size - for large dimensions like 65535, use a reasonable block size
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch kernel
    l1_normalize_kernel[grid](
        x, out, n_rows, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
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