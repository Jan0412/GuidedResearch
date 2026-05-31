import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row for input and output
    x_row_ptr = x_ptr + row_idx * dim
    out_row_ptr = out_ptr + row_idx * dim

    # Pass 1: Compute the sum of squares for the row
    sum_sq = 0.0
    i = 0
    while i < dim:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
        i += BLOCK_SIZE

    # Compute the L2 norm and its reciprocal
    norm = tl.sqrt(sum_sq)
    # Use a very small epsilon to avoid division by zero, matching PyTorch's behavior 
    # where division by zero results in NaN/Inf
    inv_norm = 1.0 / norm

    # Pass 2: Normalize the row and store the result
    i = 0
    while i < dim:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_row_ptr + offsets, x * inv_norm, mask=mask)
        i += BLOCK_SIZE

def triton_l2_norm(x: torch.Tensor):
    """
    Triton wrapper for L2 normalization along dim=1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure input is contiguous for pointer arithmetic
    x = x.contiguous()
    batch_size, dim = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)

    # Block size for processing columns
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (batch_size,)

    l2_norm_kernel[grid](
        x, 
        out, 
        dim, 
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
        Applies L2 normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        return triton_l2_norm(x)