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
    dim, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the current row
    row_ptr_x = x_ptr + row_idx * stride_x_row
    row_ptr_out = out_ptr + row_idx * stride_out_row
    
    # First pass: Compute the sum of absolute values for the row
    acc = 0.0
    for k in range(0, dim, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load data and compute absolute value
        val = tl.load(row_ptr_x + offsets, mask=mask, other=0.0)
        acc += tl.sum(tl.abs(val), axis=0)
    
    # Compute the mean (L1 norm / dim)
    mean = acc / dim
    
    # Second pass: Divide each element by the mean and store
    for k in range(0, dim, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        val = tl.load(row_ptr_x + offsets, mask=mask, other=0.0)
        # Perform normalization
        res = val / mean
        tl.store(row_ptr_out + offsets, res, mask=mask)

def triton_l1_norm(x: torch.Tensor):
    """
    Triton wrapper for L1 normalization.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    # Ensure input is contiguous for efficient pointer arithmetic
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    # Strides for row-major access
    stride_x_row = x.stride(0)
    stride_out_row = out.stride(0)
    
    # Tuning parameter: BLOCK_SIZE must be a power of 2
    BLOCK_SIZE = 1024
    
    # Grid: one program per row
    grid = (batch_size,)
    
    l1_norm_kernel[grid](
        x, out, 
        stride_x_row, stride_out_row, 
        dim, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied.
        """
        # The original operation: x / torch.mean(torch.abs(x), dim=1, keepdim=True)
        return triton_l1_norm(x)