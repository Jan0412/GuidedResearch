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
    # Get the row index this program instance handles
    row_idx = tl.program_id(0)
    
    # Compute the starting offset for this row
    row_start = row_idx * n_cols
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator for L1 norm
    l1_sum = tl.zeros([1], dtype=tl.float32)
    
    # Loop over columns in blocks
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + start_col + col_offsets
        mask = offsets < (row_start + n_cols)
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute absolute value and accumulate
        abs_x = tl.abs(x)
        l1_sum += tl.sum(abs_x, axis=0)
    
    # Compute mean = L1_sum / n_cols
    mean = l1_sum / n_cols
    
    # Second pass: normalize the data
    for start_col in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + start_col + col_offsets
        mask = offsets < (row_start + n_cols)
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Normalize: x / mean
        normalized = x / mean
        
        # Store result
        tl.store(out_ptr + offsets, normalized, mask=mask)


def triton_l1_normalize(x: torch.Tensor):
    """
    Apply L1 normalization along dimension 1 using Triton kernel.
    x / mean(abs(x), dim=1, keepdim=True)
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Get dimensions
    shape = x.shape
    n_rows = shape[0]
    n_cols = shape[1]
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Set block size - choose a good value for large dimensions
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))
    
    # Launch kernel with one block per row
    grid = (n_rows,)
    
    # Launch the Triton kernel
    l1_normalize_kernel[grid](
        x, out, n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using Triton kernels.
    """
    def __init__(self):
        """
        Initializes the optimized L1 normalization layer.
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