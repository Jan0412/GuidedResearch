import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2norm_kernel(
    x_ptr, 
    out_ptr, 
    stride_x_row, 
    stride_x_col, 
    stride_out_row, 
    stride_out_col, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row
    row_x_ptr = x_ptr + row_idx * stride_x_row
    row_out_ptr = out_ptr + row_idx * stride_out_row

    # First pass: Calculate the sum of squares for L2 norm
    sum_sq = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        # Load a block of elements from the row
        x = tl.load(row_x_ptr + offsets * stride_x_col, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute the inverse of the L2 norm
    # Adding a tiny epsilon is usually good practice, but to match torch.norm 
    # exactly, we proceed with the raw sqrt.
    inv_norm = 1.0 / tl.sqrt(sum_sq)

    # Second pass: Normalize the row and store the result
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_x_ptr + offsets * stride_x_col, mask=mask, other=0.0)
        tl.store(row_out_ptr + offsets * stride_out_col, x * inv_norm, mask=mask)

def triton_l2norm(x: torch.Tensor):
    """
    Triton wrapper for L2 normalization along dimension 1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensor is contiguous for easier indexing
    x = x.contiguous()
    rows, cols = x.shape
    
    out = torch.empty_like(x)
    
    # Strides
    stride_x_row, stride_x_col = x.stride()
    stride_out_row, stride_out_col = out.stride()
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (rows,)
    
    l2norm_kernel[grid](
        x, out, 
        stride_x_row, stride_x_col, 
        stride_out_row, stride_out_col, 
        cols, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor along dim=1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        # Replace x / torch.norm(x, p=2, dim=1, keepdim=True) with Triton kernel
        return triton_l2norm(x)