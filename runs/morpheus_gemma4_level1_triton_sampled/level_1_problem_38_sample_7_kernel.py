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
    pid = tl.program_id(0)
    
    # Pointers to the start of the current row
    row_x_ptr = x_ptr + pid * stride_x_row
    row_out_ptr = out_ptr + pid * stride_out_row
    
    # First pass: Compute the sum of absolute values for the row
    row_sum = 0.0
    num_blocks = tl.cdiv(dim, BLOCK_SIZE)
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_val = tl.load(row_x_ptr + offsets, mask=mask, other=0.0)
        row_sum += tl.sum(tl.abs(x_val), axis=0)
        
    # L1 mean = sum(|x|) / dim
    # We want x / mean = x / (sum / dim) = (x * dim) / sum
    inv_mean = dim / row_sum
    
    # Second pass: Multiply each element by the inverse mean
    for i in range(num_blocks):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_val = tl.load(row_x_ptr + offsets, mask=mask, other=0.0)
        tl.store(row_out_ptr + offsets, x_val * inv_mean, mask=mask)

def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper for L1 normalization along dimension 1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    
    batch, dim = x.shape
    out = torch.empty_like(x)
    
    stride_x_row = x.stride(0)
    stride_out_row = out.stride(0)
    
    # BLOCK_SIZE is chosen to be a power of 2 for Triton efficiency
    BLOCK_SIZE = 1024
    grid = (batch,)
    
    l1_norm_kernel[grid](
        x, 
        out, 
        stride_x_row, 
        stride_out_row, 
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
        Applies L1 normalization to the input tensor using a Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, dim).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied.
        """
        return triton_l1_norm(x)