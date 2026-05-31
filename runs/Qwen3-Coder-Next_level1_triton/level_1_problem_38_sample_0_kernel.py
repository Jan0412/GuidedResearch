import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_normalize_kernel(
    x_ptr,  # Input tensor
    out_ptr,  # Output tensor
    n_rows,  # Number of rows (batch_size)
    n_cols,  # Number of columns (dim)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute the mean for this row
    row_start = row_idx * n_cols
    sum_abs = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over columns in blocks
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load data
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        # Compute absolute value and accumulate
        abs_x = tl.abs(x)
        sum_abs += tl.sum(abs_x, axis=0)
    
    # Compute mean (divide by n_cols)
    mean = sum_abs / n_cols
    
    # Normalize the row by dividing by the mean
    inv_mean = 1.0 / mean
    
    # Store normalized values
    for col_start in range(0, n_cols, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load data
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        # Normalize
        normalized = x * inv_mean
        # Store result
        tl.store(out_ptr + row_start + offsets, normalized, mask=mask)


def triton_l1_normalize(x: torch.Tensor):
    """
    Applies L1 normalization to the input tensor using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    original_shape = x.shape
    if x.dim() > 2:
        # Reshape to 2D if needed: (batch_size, dim)
        x = x.view(-1, x.shape[-1])
    
    batch_size, dim = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size (tunable)
    BLOCK_SIZE = 1024
    
    # Grid: one block per row
    grid = (batch_size,)
    
    # Launch kernel
    l1_normalize_kernel[grid](
        x, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape back to original shape if needed
    out = out.view(original_shape)
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