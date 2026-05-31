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
    # Each program handles one row (batch dimension)
    batch_idx = tl.program_id(0)
    
    # Shared memory for reduction
    shared_mean = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    if batch_idx < batch_size:
        # Load input data for this batch
        x_row = tl.load(x_ptr + batch_idx * dim + tl.arange(0, BLOCK_SIZE), 
                       mask=tl.arange(0, BLOCK_SIZE) < dim, other=0.0)
        
        # Compute absolute values
        abs_x = tl.abs(x_row)
        
        # Compute mean using reduction
        mean_val = tl.sum(abs_x, axis=0) / dim
        
        # Store mean to global memory
        tl.store(mean_ptr + batch_idx, mean_val)
        
        # Normalize the row
        normalized = x_row / mean_val
        
        # Store result
        tl.store(out_ptr + batch_idx * dim + tl.arange(0, BLOCK_SIZE),
                normalized, mask=tl.arange(0, BLOCK_SIZE) < dim)

@triton.jit
def l1_norm_reduce_kernel(
    x_ptr,
    mean_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch dimension)
    batch_idx = tl.program_id(0)
    
    if batch_idx < batch_size:
        # Load input data for this batch
        x_row = tl.load(x_ptr + batch_idx * dim + tl.arange(0, BLOCK_SIZE), 
                       mask=tl.arange(0, BLOCK_SIZE) < dim, other=0.0)
        
        # Compute absolute values
        abs_x = tl.abs(x_row)
        
        # Compute mean using reduction
        mean_val = tl.sum(abs_x, axis=0) / dim
        
        # Store mean to global memory
        tl.store(mean_ptr + batch_idx, mean_val)

@triton.jit
def l1_norm_normalize_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch dimension)
    batch_idx = tl.program_id(0)
    
    if batch_idx < batch_size:
        # Load mean value
        mean_val = tl.load(mean_ptr + batch_idx)
        
        # Load input data for this batch
        x_row = tl.load(x_ptr + batch_idx * dim + tl.arange(0, BLOCK_SIZE), 
                       mask=tl.arange(0, BLOCK_SIZE) < dim, other=0.0)
        
        # Normalize the row
        normalized = x_row / mean_val
        
        # Store result
        tl.store(out_ptr + batch_idx * dim + tl.arange(0, BLOCK_SIZE),
                normalized, mask=tl.arange(0, BLOCK_SIZE) < dim)

def triton_l1_norm(x: torch.Tensor):
    """
    Triton-based L1 normalization implementation
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
    
    # First kernel: compute means
    grid_means = (batch_size, 1)
    l1_norm_reduce_kernel[grid_means](
        x, means, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: normalize
    grid_norm = (batch_size, 1)
    l1_norm_normalize_kernel[grid_norm](
        x, means, out, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
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