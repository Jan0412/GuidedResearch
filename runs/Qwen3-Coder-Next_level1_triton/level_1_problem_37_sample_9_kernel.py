import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def frobenius_norm_kernel(
    x_ptr,
    n_elements,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Global thread ID
    pid = tl.program_id(0)
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for out-of-bounds elements
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared values
    x_sq = x * x
    
    # Use a simple reduction approach with shared memory
    # We'll accumulate partial sums in a register first
    acc = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, BLOCK_SIZE, 32):
        indices = offsets + i
        masks = indices < n_elements
        vals = tl.load(x_ptr + indices, mask=masks, other=0.0)
        acc += tl.sum(vals * vals)
    
    # Store the partial sum for this block
    tl.store(out_ptr + pid, acc)


@triton.jit
def frobenius_norm_reduction_kernel(
    partial_sums_ptr,
    n_partial_sums,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Global thread ID
    pid = tl.program_id(0)
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for out-of-bounds elements
    mask = offsets < n_partial_sums
    
    # Load data and accumulate
    acc = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, BLOCK_SIZE, 32):
        indices = offsets + i
        masks = indices < n_partial_sums
        vals = tl.load(partial_sums_ptr + indices, mask=masks, other=0.0)
        acc += vals
    
    # Store result
    tl.store(out_ptr + pid, acc)


@triton.jit
def frobenius_norm_final_reduction_kernel(
    partial_sums_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Global thread ID
    pid = tl.program_id(0)
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for out-of-bounds elements
    mask = offsets < BLOCK_SIZE
    
    # Load data and accumulate
    acc = tl.zeros((1,), dtype=tl.float32)
    vals = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    acc = tl.sum(vals)
    
    # Store result
    tl.store(out_ptr, acc)


@triton.jit
def frobenius_normalize_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Global thread ID
    pid = tl.program_id(0)
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for out-of-bounds elements
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load norm (scalar)
    norm = tl.load(norm_ptr)
    
    # Normalize
    out = x / norm
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor):
    """
    Computes the Frobenius norm of a tensor using Triton kernels.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    n_elements = x.numel()
    
    # First kernel: compute partial sums in parallel
    BLOCK_SIZE = 256
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # If the number of blocks is large, we might need multiple reduction steps
    partial_sums = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
    
    grid = (num_blocks,)
    frobenius_norm_kernel[grid](x, n_elements, partial_sums, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reduce partial sums until we get a single value
    while num_blocks > 1:
        BLOCK_SIZE_REDUCTION = 256
        new_num_blocks = (num_blocks + BLOCK_SIZE_REDUCTION - 1) // BLOCK_SIZE_REDUCTION
        new_partial_sums = torch.empty(new_num_blocks, device=x.device, dtype=torch.float32)
        
        grid = (new_num_blocks,)
        frobenius_norm_reduction_kernel[grid](partial_sums, num_blocks, new_partial_sums, BLOCK_SIZE=BLOCK_SIZE_REDUCTION)
        
        partial_sums = new_partial_sums
        num_blocks = new_num_blocks
    
    # Final reduction to get scalar norm
    norm = torch.empty(1, device=x.device, dtype=torch.float32)
    grid = (1,)
    frobenius_norm_final_reduction_kernel[grid](partial_sums, norm, BLOCK_SIZE=256)
    
    return norm


def triton_frobenius_normalize(x: torch.Tensor):
    """
    Applies Frobenius norm normalization to the input tensor.
    """
    norm = triton_frobenius_norm(x)
    
    x = x.contiguous()
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    frobenius_normalize_kernel[grid](x, norm, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return triton_frobenius_normalize(x)