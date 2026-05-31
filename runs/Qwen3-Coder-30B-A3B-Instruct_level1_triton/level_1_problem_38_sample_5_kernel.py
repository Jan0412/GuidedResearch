import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute row index
    row_idx = tl.program_id(0)
    
    if row_idx >= batch_size:
        return
    
    # Compute pointers for this row
    x_row_ptr = x_ptr + row_idx * dim
    out_row_ptr = out_ptr + row_idx * dim
    mean_ptr_row = mean_ptr + row_idx
    
    # Reduce to compute mean absolute value
    mean = 0.0
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(x_vals)
        mean += tl.sum(abs_vals)
    
    # Normalize mean across all elements in the row
    mean = mean / dim
    tl.store(mean_ptr_row, mean)
    
    # Compute normalized output
    for i in range(0, dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim
        x_vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        mean_val = mean
        out_vals = x_vals / mean_val
        tl.store(out_row_ptr + offsets, out_vals, mask=mask)

def triton_l1_norm(x: torch.Tensor):
    """
    Triton-based L1 normalization implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Allocate mean tensor
    means = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    
    # Configure block size
    BLOCK_SIZE = 1024
    
    # Grid configuration
    grid = lambda meta: (batch_size,)
    
    # Launch kernel
    l1_norm_kernel[grid](
        x,
        out,
        means,
        batch_size,
        dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernel for L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return triton_l1_norm(x)