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
    
    # Load input row
    row_start = batch_idx * dim
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in chunks
    for i in range(0, dim, BLOCK_SIZE):
        # Calculate actual offsets
        chunk_offsets = row_start + i + offsets
        mask = chunk_offsets < (row_start + dim)
        
        # Load data
        x_vals = tl.load(x_ptr + chunk_offsets, mask=mask, other=0.0)
        
        # Compute absolute values
        abs_vals = tl.abs(x_vals)
        
        # Store in shared memory for reduction
        tl.store(shared_mean + offsets, abs_vals, mask=mask)
        
        # Synchronize threads before reduction
        tl.sync()
        
        # Reduce within block
        if i == 0:
            # First iteration - initialize mean
            local_mean = tl.sum(shared_mean, axis=0)
        else:
            # Subsequent iterations - accumulate
            local_mean += tl.sum(shared_mean, axis=0)
    
    # Write mean to global memory
    if batch_idx < batch_size:
        mean_val = local_mean / dim
        tl.store(mean_ptr + batch_idx, mean_val)
    
    # Now compute normalized output
    # Load mean for this batch
    mean = tl.load(mean_ptr + batch_idx)
    
    # Compute normalized values
    for i in range(0, dim, BLOCK_SIZE):
        chunk_offsets = row_start + i + offsets
        mask = chunk_offsets < (row_start + dim)
        
        # Load original values
        x_vals = tl.load(x_ptr + chunk_offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = x_vals / mean
        
        # Store results
        tl.store(out_ptr + chunk_offsets, normalized, mask=mask)

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
    
    # Shared memory for reduction
    shared_abs = tl.shared_memory(dtype=tl.float32, shape=(BLOCK_SIZE,))
    
    # Load input row
    row_start = batch_idx * dim
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    local_sum = 0.0
    
    # Process in chunks
    for i in range(0, dim, BLOCK_SIZE):
        # Calculate actual offsets
        chunk_offsets = row_start + i + offsets
        mask = chunk_offsets < (row_start + dim)
        
        # Load data
        x_vals = tl.load(x_ptr + chunk_offsets, mask=mask, other=0.0)
        
        # Compute absolute values
        abs_vals = tl.abs(x_vals)
        
        # Store in shared memory for reduction
        tl.store(shared_abs + offsets, abs_vals, mask=mask)
        
        # Synchronize threads before reduction
        tl.sync()
        
        # Reduce within block
        local_sum += tl.sum(shared_abs, axis=0)
    
    # Write mean to global memory
    if batch_idx < batch_size:
        mean_val = local_sum / dim
        tl.store(mean_ptr + batch_idx, mean_val)

@triton.jit
def l1_norm_normalize_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (batch element)
    batch_idx = tl.program_id(0)
    
    # Load mean for this batch
    mean = tl.load(mean_ptr + batch_idx)
    
    # Load input row
    row_start = batch_idx * dim
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in chunks
    for i in range(0, dim, BLOCK_SIZE):
        # Calculate actual offsets
        chunk_offsets = row_start + i + offsets
        mask = chunk_offsets < (row_start + dim)
        
        # Load original values
        x_vals = tl.load(x_ptr + chunk_offsets, mask=mask, other=0.0)
        
        # Normalize
        normalized = x_vals / mean
        
        # Store results
        tl.store(out_ptr + chunk_offsets, normalized, mask=mask)

def triton_l1_norm(x: torch.Tensor):
    """
    Triton implementation of L1 normalization using separate kernels for mean computation
    and normalization.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, dim = x.shape
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Compute means
    means = torch.empty(batch_size, dtype=torch.float32, device='cuda')
    
    # Grid configuration
    BLOCK_SIZE = 1024
    grid_mean = (batch_size,)
    
    # Compute means first
    l1_norm_mean_kernel[grid_mean](
        x, means, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Then normalize
    grid_norm = (batch_size,)
    l1_norm_normalize_kernel[grid_norm](
        x, out, means, batch_size, dim, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized model that performs L1 normalization using Triton kernels.
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