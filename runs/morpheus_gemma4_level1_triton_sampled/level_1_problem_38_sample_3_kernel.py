import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr, 
    out_ptr, 
    stride_x_row, 
    stride_out_row, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform L1 normalization along the second dimension.
    For each row:
      1. Compute the sum of absolute values of all elements.
      2. Calculate the mean (sum / n_cols).
      3. Divide every element in the row by this mean.
    """
    # Each program handles one row (batch element)
    row_idx = tl.program_id(0)
    
    # Row pointers
    row_x_ptr = x_ptr + row_idx * stride_x_row
    row_out_ptr = out_ptr + row_idx * stride_out_row
    
    # --- Pass 1: Compute sum of absolute values ---
    sum_abs = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        # Load elements and compute absolute values
        vals = tl.load(row_x_ptr + cols, mask=mask, other=0.0)
        sum_abs += tl.sum(tl.abs(vals), axis=0)
    
    # Calculate the inverse of the mean to replace division with multiplication
    # mean = sum_abs / n_cols
    # inv_mean = 1.0 / (sum_abs / n_cols) = n_cols / sum_abs
    inv_mean = n_cols / sum_abs
    
    # --- Pass 2: Multiply by inverse mean and store ---
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(row_x_ptr + cols, mask=mask, other=0.0)
        # Apply normalization
        normalized_vals = vals * inv_mean
        tl.store(row_out_ptr + cols, normalized_vals, mask=mask)


def triton_l1_norm(x: torch.Tensor):
    """
    Wrapper for the L1 normalization Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure tensor is contiguous for efficient memory access
    x = x.contiguous()
    n_rows, n_cols = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Strides for row-wise access
    stride_x_row = x.stride(0)
    stride_out_row = out.stride(0)
    
    # Tuning parameter: Block size for loading elements
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    grid = (n_rows,)
    
    # Launch kernel
    l1_norm_kernel[grid](
        x, 
        out, 
        stride_x_row, 
        stride_out_row, 
        n_cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using a custom Triton kernel.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied.
        """
        return triton_l1_norm(x)