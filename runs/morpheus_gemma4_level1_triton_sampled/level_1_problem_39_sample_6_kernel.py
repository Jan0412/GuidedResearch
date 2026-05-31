import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr, 
    out_ptr,
    stride_x_row, 
    stride_x_col,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform L2 normalization along the second dimension (rows).
    Each program handles one row.
    """
    # Get the row index this program is responsible for
    row_idx = tl.program_id(0)
    
    # Pointer to the start of the row for input and output
    # We assume out has the same layout as x
    row_x_ptr = x_ptr + row_idx * stride_x_row
    row_out_ptr = out_ptr + row_idx * stride_x_row
    
    # First pass: Calculate the sum of squares for the row
    sum_sq = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load a block of the row
        x = tl.load(row_x_ptr + offsets * stride_x_col, mask=mask, other=0.0)
        # Accumulate sum of squares
        sum_sq += tl.sum(x * x, axis=0)
        
    # Compute the L2 norm and its reciprocal
    norm = tl.sqrt(sum_sq)
    inv_norm = 1.0 / norm
    
    # Second pass: Divide each element by the norm and store the result
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load the block again
        x = tl.load(row_x_ptr + offsets * stride_x_col, mask=mask, other=0.0)
        # Normalize and store
        tl.store(row_out_ptr + offsets * stride_x_col, x * inv_norm, mask=mask)


def triton_l2_norm(x: torch.Tensor):
    """
    Wrapper for the Triton L2 normalization kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure the tensor is contiguous to simplify stride calculations
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    stride_x_row = x.stride(0)
    stride_x_col = x.stride(1)
    
    # Tunable parameter for block size
    BLOCK_SIZE = 1024
    # Grid is one program per row
    grid = (n_rows, )
    
    l2_norm_kernel[grid](
        x, out,
        stride_x_row, stride_x_col,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using a custom Triton kernel.
    """
    def __init__(self):
        """
        Initializes the ModelNew layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        return triton_l2_norm(x)