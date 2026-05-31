import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    stride_row,     # Stride between rows
    dim,            # Number of elements per row
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Pointers for the start of the current row
    row_x_ptr = x_ptr + row_idx * stride_row
    row_out_ptr = out_ptr + row_idx * stride_row

    # Pass 1: Compute the sum of absolute values for the row
    abs_sum = 0.0
    for k in range(0, tl.cdiv(dim, BLOCK_SIZE)):
        offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        # Load values, using 0.0 for out-of-bounds elements to not affect the sum
        vals = tl.load(row_x_ptr + offsets, mask=mask, other=0.0)
        abs_sum += tl.sum(tl.abs(vals), axis=0)

    # Compute the mean (L1 norm / dim)
    mean = abs_sum / dim

    # Pass 2: Divide each element by the mean and store the result
    for k in range(0, tl.cdiv(dim, BLOCK_SIZE)):
        offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        vals = tl.load(row_x_ptr + offsets, mask=mask, other=0.0)
        # Perform the division (equivalent to x / mean)
        # If mean is 0, this will result in Inf/NaN, matching PyTorch behavior
        res = vals / mean
        tl.store(row_out_ptr + offsets, res, mask=mask)

def triton_l1_norm(x: torch.Tensor):
    """
    Triton wrapper for L1 normalization.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    out = torch.empty_like(x)
    
    stride_row = x.stride(0)
    
    # Use a reasonable BLOCK_SIZE for FP32
    BLOCK_SIZE = 1024
    
    # Grid is one program per row
    grid = (batch_size,)
    
    l1_norm_kernel[grid](
        x, 
        out, 
        stride_row, 
        dim, 
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
        Applies L1 normalization to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied.
        """
        # The original operation: x / torch.mean(torch.abs(x), dim=1, keepdim=True)
        return triton_l1_norm(x)