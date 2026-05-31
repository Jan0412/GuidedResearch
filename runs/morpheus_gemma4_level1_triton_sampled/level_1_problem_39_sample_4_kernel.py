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
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * stride_row
    out_row_ptr = out_ptr + row_idx * stride_row
    
    # Pass 1: Compute the sum of squares for the row
    sum_sq = 0.0
    i = 0
    while i < dim:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        sum_sq += tl.sum(vals * vals)
        i += BLOCK_SIZE
    
    # Compute the reciprocal of the L2 norm
    inv_norm = 1.0 / tl.sqrt(sum_sq)
    
    # Pass 2: Normalize each element and store the result
    i = 0
    while i < dim:
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(row_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_row_ptr + offsets, vals * inv_norm, mask=mask)
        i += BLOCK_SIZE

def triton_l2_norm(x: torch.Tensor):
    """
    Triton wrapper for L2 normalization along the second dimension.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure the tensor is contiguous for efficient pointer arithmetic
    x = x.contiguous()
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    # Tunable block size for the dimension reduction
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (batch_size,)
    
    # Launch the kernel
    l2_norm_kernel[grid](
        x, 
        out, 
        x.stride(0), 
        dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L2 normalization using custom Triton kernels.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied.
        """
        return triton_l2_norm(x)