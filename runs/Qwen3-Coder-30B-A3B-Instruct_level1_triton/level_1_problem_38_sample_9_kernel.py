import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_mean_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output
    pid = tl.program_id(0)
    batch_idx = pid // dim
    elem_idx = pid % dim
    
    if batch_idx >= batch_size:
        return
        
    # Calculate the starting position for this batch
    batch_start = batch_idx * dim
    
    # Load the absolute values for this element across the dimension
    abs_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process in chunks to handle large dimensions
    for i in range(0, dim, BLOCK_SIZE):
        offsets = batch_start + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < batch_start + dim
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        abs_vals = tl.where(mask, tl.abs(vals), abs_vals)
        
    # Reduce to compute sum for this batch
    sum_val = tl.sum(abs_vals, axis=0)
    
    # Store the mean for this batch
    mean_val = sum_val / dim
    tl.store(out_ptr + batch_idx, mean_val)

@triton.jit
def l1_norm_divide_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    n_elements,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output
    pid = tl.program_id(0)
    batch_idx = pid // dim
    elem_idx = pid % dim
    
    if batch_idx >= batch_size:
        return
        
    # Load input value
    x_val = tl.load(x_ptr + pid)
    
    # Load corresponding mean
    mean_val = tl.load(mean_ptr + batch_idx)
    
    # Perform division
    out_val = x_val / mean_val
    
    # Store result
    tl.store(out_ptr + pid, out_val)

def triton_l1_norm(x: torch.Tensor):
    """
    Triton implementation of L1 normalization using two kernels:
    1. Compute mean per batch
    2. Divide each element by its batch mean
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    n_elements = batch_size * dim
    
    # Allocate memory for means
    means = torch.empty(batch_size, dtype=torch.float32, device=x.device)
    
    # Kernel 1: Compute means
    BLOCK_SIZE = 1024
    grid1 = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    l1_norm_mean_kernel[grid1](
        x, means, n_elements, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Kernel 2: Divide by means
    out = torch.empty_like(x)
    grid2 = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    l1_norm_divide_kernel[grid2](
        x, means, out, n_elements, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernels for L1 normalization.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        return triton_l1_norm(x)