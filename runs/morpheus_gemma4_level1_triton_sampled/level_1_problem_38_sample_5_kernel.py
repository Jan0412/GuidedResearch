import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr, 
    out_ptr, 
    stride_row, 
    stride_col, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Pointers for the current row
    row_x_ptr = x_ptr + row_idx * stride_row
    row_out_ptr = out_ptr + row_idx * stride_row
    
    # Create offsets for the columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load the row data
    x = tl.load(row_x_ptr + col_offsets * stride_col, mask=mask, other=0.0)
    
    # Compute the L1 mean: mean(|x|) = (sum(|x|)) / n_cols
    abs_x = tl.abs(x)
    # Sum the absolute values across the row
    # We use a mask in the sum as well by ensuring out-of-bounds are 0.0
    sum_abs = tl.sum(abs_x, axis=0)
    mean_abs = sum_abs / n_cols
    
    # Perform the normalization: x / mean_abs
    # To avoid division by zero, we follow PyTorch's behavior (results in inf/nan)
    out = x / mean_abs
    
    # Store the result back to memory
    tl.store(row_out_ptr + col_offsets * stride_col, out, mask=mask)

def triton_l1_norm(x: torch.Tensor):
    """
    Triton wrapper for L1 normalization along dim=1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    
    # Ensure tensor is contiguous for pointer arithmetic
    x = x.contiguous()
    M, N = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Find the smallest power of 2 >= N for the BLOCK_SIZE
    # Since dim is 65535, 2^16 = 65536 is the appropriate block size.
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Grid is one program per row
    grid = (M,)
    
    l1_norm_kernel[grid](
        x, 
        out, 
        x.stride(0), 
        x.stride(1), 
        N, 
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
        Applies L1 normalization to the input tensor using the Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied.
        """
        return triton_l1_norm(x)