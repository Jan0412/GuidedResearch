import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_normalize_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for L1 normalization: x / mean(|x|, dim=1)
    
    Each program handles one row of the input tensor.
    """
    # Row index
    row_idx = tl.program_id(0)
    
    # Calculate the starting offset for this row
    row_start = row_idx * n_cols
    
    # Accumulator for L1 norm (sum of absolute values)
    l1_sum = tl.zeros([1], dtype=tl.float32)
    
    # Process columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Compute absolute value and accumulate
        abs_x = tl.abs(x)
        l1_sum += tl.sum(abs_x, axis=0)
    
    # Compute mean: mean = sum / n_cols
    mean = l1_sum / n_cols
    
    # Normalize: x / mean
    for start_col in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start_col + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load data
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = x / mean
        
        # Store result
        tl.store(out_ptr + row_start + col_offsets, normalized, mask=mask)


def triton_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L1 normalization to the input tensor along dimension 1.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
        
    Returns:
        Output tensor with L1 normalization applied, same shape as input.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    assert x.dim() == 2, "Input must be 2D tensor."
    
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    
    # Use a reasonable block size
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the kernel
    l1_normalize_kernel[grid](
        x,
        out,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using custom Triton kernel.
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