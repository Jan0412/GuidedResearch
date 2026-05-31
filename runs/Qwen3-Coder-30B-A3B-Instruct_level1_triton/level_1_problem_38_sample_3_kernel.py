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
    # Each program handles one row (batch element)
    batch_idx = tl.program_id(0)
    
    # Shared memory for reduction
    shared_mean = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    if batch_idx < batch_size:
        # Compute mean of absolute values for this batch element
        mean = 0.0
        for i in range(0, dim, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < dim
            x_vals = tl.load(x_ptr + batch_idx * dim + offs, mask=mask, other=0.0)
            abs_vals = tl.abs(x_vals)
            mean += tl.sum(abs_vals)
        
        # Reduce across the dimension
        mean = mean / dim
        
        # Store mean
        tl.store(mean_ptr + batch_idx, mean)
        
        # Compute normalized output
        for i in range(0, dim, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < dim
            x_vals = tl.load(x_ptr + batch_idx * dim + offs, mask=mask, other=0.0)
            mean_val = mean
            out_vals = x_vals / mean_val
            tl.store(out_ptr + batch_idx * dim + offs, out_vals, mask=mask)

@triton.jit
def l1_norm_mean_kernel(
    x_ptr,
    mean_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element)
    batch_idx = tl.program_id(0)
    
    if batch_idx < batch_size:
        # Compute mean of absolute values for this batch element
        mean = 0.0
        for i in range(0, dim, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < dim
            x_vals = tl.load(x_ptr + batch_idx * dim + offs, mask=mask, other=0.0)
            abs_vals = tl.abs(x_vals)
            mean += tl.sum(abs_vals)
        
        # Reduce across the dimension
        mean = mean / dim
        
        # Store mean
        tl.store(mean_ptr + batch_idx, mean)

@triton.jit
def l1_norm_divide_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element)
    batch_idx = tl.program_id(0)
    
    if batch_idx < batch_size:
        # Load mean value
        mean_val = tl.load(mean_ptr + batch_idx)
        
        # Compute normalized output
        for i in range(0, dim, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            mask = offs < dim
            x_vals = tl.load(x_ptr + batch_idx * dim + offs, mask=mask, other=0.0)
            out_vals = x_vals / mean_val
            tl.store(out_ptr + batch_idx * dim + offs, out_vals, mask=mask)

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
    means = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # First kernel: compute means
    grid_mean = lambda meta: (batch_size,)
    l1_norm_mean_kernel[grid_mean](x, means, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second kernel: compute division
    grid_div = lambda meta: (batch_size,)
    l1_norm_divide_kernel[grid_div](x, means, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model with Triton-based L1 normalization.
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