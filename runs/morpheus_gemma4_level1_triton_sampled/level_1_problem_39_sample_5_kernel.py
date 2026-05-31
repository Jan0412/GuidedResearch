import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr, 
    out_ptr, 
    stride_row, 
    dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row
    row_start_ptr = x_ptr + row_idx * stride_row
    out_start_ptr = out_ptr + row_idx * stride_row

    # 1. Compute the sum of squares for the row
    sum_sq = 0.0
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load values and compute square
        vals = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(vals * vals, axis=0)

    # Compute the L2 norm
    norm = tl.sqrt(sum_sq)
    # Prevent division by zero
    norm = tl.maximum(norm, 1e-12)

    # 2. Normalize and store the results
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        # Divide by the computed norm and store
        tl.store(out_start_ptr + offsets, vals / norm, mask=mask)

def triton_l2_norm(x: torch.Tensor):
    """
    Triton wrapper for L2 normalization along dimension 1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure tensor is contiguous to simplify pointer arithmetic
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    stride_row = x.stride(0)
    BLOCK_SIZE = 1024  # Tunable parameter for block size
    
    # Grid is one program per row
    grid = (batch_size,)
    
    l2_norm_kernel[grid](
        x, out, stride_row, dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return triton_l2_norm(x)