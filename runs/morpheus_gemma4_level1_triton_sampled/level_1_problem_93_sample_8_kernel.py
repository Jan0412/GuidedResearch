import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def masked_cumsum_kernel(
    x_ptr, mask_ptr, out_ptr,
    n_rows, n_cols,
    stride_x_row, stride_x_col,
    stride_m_row, stride_m_col,
    stride_out_row, stride_out_col,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform masked cumulative sum.
    Fuses the element-wise multiplication (x * mask) and the cumsum operation.
    """
    if dim == 1:
        # Process one row per program
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        # Load x and mask for the row
        x = tl.load(x_ptr + row_idx * stride_x_row + col_offsets * stride_x_col, mask=mask, other=0.0)
        m = tl.load(mask_ptr + row_idx * stride_m_row + col_offsets * stride_m_col, mask=mask, other=False)
        
        # Masked values: replace False with 0.0
        val = tl.where(m, x, 0.0)
        # Perform cumulative sum along the row
        res = tl.cumsum(val, axis=0)
        
        # Store result
        tl.store(out_ptr + row_idx * stride_out_row + col_offsets * stride_out_col, res, mask=mask)
    else:
        # Process one column per program
        col_idx = tl.program_id(0)
        row_offsets = tl.arange(0, BLOCK_SIZE)
        mask = row_offsets < n_rows
        
        # Load x and mask for the column
        x = tl.load(x_ptr + row_offsets * stride_x_row + col_idx * stride_x_col, mask=mask, other=0.0)
        m = tl.load(mask_ptr + row_offsets * stride_m_row + col_idx * stride_m_col, mask=mask, other=False)
        
        # Masked values: replace False with 0.0
        val = tl.where(m, x, 0.0)
        # Perform cumulative sum along the column
        res = tl.cumsum(val, axis=0)
        
        # Store result
        tl.store(out_ptr + row_offsets * stride_out_row + col_idx * stride_out_col, res, mask=mask)

def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int):
    assert x.is_cuda and mask.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensors are contiguous for simpler stride calculation
    x = x.contiguous()
    mask = mask.contiguous()
    
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    # Determine which dimension we are summing over
    # If dim=1, we sum across columns (row-wise). Grid is number of rows.
    # If dim=0, we sum across rows (column-wise). Grid is number of columns.
    if dim == 1:
        grid = (n_rows,)
        sum_dim_size = n_cols
    else:
        grid = (n_cols,)
        sum_dim_size = n_rows

    # BLOCK_SIZE must be a power of 2 and >= the size of the summing dimension.
    # For the given input shape (32768), 32768 is 2^15.
    BLOCK_SIZE = triton.next_power_of_2(sum_dim_size)

    masked_cumsum_kernel[grid](
        x, mask, out,
        n_rows, n_cols,
        x.stride(0), x.stride(1),
        mask.stride(0), mask.stride(1),
        out.stride(0), out.stride(1),
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized version of the Model that performs a masked cumulative sum
    using a custom fused Triton kernel.
    """
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        return triton_masked_cumsum(x, mask, self.dim)