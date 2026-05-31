import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_normalize_kernel(
    x_ptr,  # Input pointer
    y_ptr,  # Output pointer
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (dim)
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate row start offset
    row_start = row_idx * n_cols
    
    # Create column offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load row data
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    
    # Compute absolute values and sum them
    abs_x = tl.abs(x)
    row_sum = tl.sum(abs_x, axis=0)
    
    # Compute mean: sum / n_cols
    mean = row_sum / n_cols
    
    # Load original x again and divide by mean
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    normalized = x / mean
    
    # Store result
    tl.store(y_ptr + row_start + offsets, normalized, mask=mask)


def triton_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Applies L1 normalization to the input tensor using Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        
    Returns:
        torch.Tensor: Output tensor with L1 normalization applied
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    assert x.dim() == 2, "Input must be 2D tensor for this implementation"
    
    x = x.contiguous()
    y = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    
    # Use a reasonable block size, tuned for the dimension size
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = (n_rows,)
    
    # Launch the kernel
    l1_normalize_kernel[grid](
        x, y, n_rows, n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


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